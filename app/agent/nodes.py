import re
import sqlite3
import pandas as pd
import logging
from typing import Literal
from langchain_core.prompts import PromptTemplate
from langgraph.types import Command

from app.core.cache import chroma_client, cache_collection
from app.core.state import AgentActionState, ActionClassification, SourceClassification
from app.core.config import DB_PATH
from app.db.database import get_dynamic_schema
from app.agent.llm import reasoning_llm, coder_llm

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def classify_intent(state: AgentActionState) -> Command[Literal["classify_sources", "draft_response"]]:
    logger.info("=== NODE: classify_intent ===")
    logger.info(f"Incoming State User Request: {state.get('user_request')}")

    structured_text_llm = reasoning_llm.with_structured_output(ActionClassification)

    classification_prompt = f"""
        You are an intelligent routing agent for an Energy Management System.
        Analyze the user's message and classify its primary intent based on the following strict rules:

        INTENT CATEGORIES:
        - 'data_add': The user explicitly provides a file path (e.g., .csv) or asks to ingest new data.
        - 'data_modify': The user asks to UPDATE, DELETE, or correct existing grid/energy data.
        - 'query': The user asks a specific, targeted question (e.g., "What was the peak load yesterday?", "Did substation A go down?").
        - 'report': The user asks for a broader analysis, trend summary, or comprehensive overview.
        - 'irrelevant': The prompt has absolutely nothing to do with energy grids, power supply, outages, or data analysis.

        User's request: {state['user_request']}
        """

    logger.info(f"LLM Prompt (classify_intent):\n{classification_prompt}")
    classification = structured_text_llm.invoke(classification_prompt)
    logger.info(f"LLM Output (Classification): {classification}")

    goto = "draft_response" if classification["intent"] == "irrelevant" else "classify_sources"
    logger.info(f"Routing to: {goto}")
    return Command(goto=goto, update={"classification": classification})


def classify_sources(state: AgentActionState) -> Command[Literal["data_agent"]]:
    logger.info("=== NODE: classify_sources ===")
    logger.info(f"Current Intent: {state['classification'].get('intent')}")

    structured_text_llm = reasoning_llm.with_structured_output(SourceClassification)

    classifications_prompt = f"""
        You must determine the correct data source required to answer the user's energy query.

        SOURCES:
        1. "outage_reports": Unstructured text data. 
           -> Select this if the user asks about: grid faults, equipment failures, blackouts, weather damage, repair logs.
        2. "demand_reports": Structured telemetry/time-series data. 
           -> Select this if the user asks about: MW/h consumption, peak load, generation capacity, voltage levels.
        3. "both": 
           -> Select this if the query requires correlating an event with data.

        User's query: {state['user_request']}
        Intent Summary: {state["classification"].get("summary", "Unknown")}
        """

    logger.info(f"LLM Prompt (classify_sources):\n{classifications_prompt}")
    classification = structured_text_llm.invoke(classifications_prompt)
    logger.info(f"LLM Output (Source Classification): {classification}")

    intent = state["classification"]["intent"]

    if intent in ["data_add", "data_modify", "query", "report"]:
        goto = "data_agent"
    else:
        goto = "report_agent"

    logger.info(f"Routing to: {goto}")
    return Command(goto=goto, update={"source_classification": classification})


def data_agent(state: AgentActionState) -> Command[Literal["analysis_agent", "report_agent"]]:
    logger.info("=== NODE: data_agent ===")
    intent = state["classification"]["intent"]
    user_request = state["user_request"]
    updates = {}

    if intent == "data_add":
        match = re.search(r'([\w\-./\\]+\.csv)', user_request)
        if match:
            csv_file = match.group(1)
            logger.info(f"Detected CSV file for ingestion: {csv_file}")
            try:
                df = pd.read_csv(csv_file)
                columns = ", ".join(df.columns)
                data_sample = df.head(1).to_markdown(index=False)

                validation_prompt = PromptTemplate.from_template("""
                        You are a Data Architect for a power grid operator. 
                        Analyze the following CSV columns and sample data.
                        Columns: {columns}
                        Data Sample:
                        {sample}

                        Determine if this data is a fit for:
                        1. 'outage_reports' (grid faults, equipment failures, blackouts, weather damage, repair logs)
                        2. 'demand_reports' (telemetry, MW/h consumption, peak load, generation capacity, voltage levels)
                        3. 'neither' (irrelevant data not matching the two categories above)

                        CRITICAL: Output ONLY one of these three exact words: outage_reports, demand_reports, or neither.
                        """)

                logger.info(
                    f"LLM Prompt (Data Validation):\n{validation_prompt.format(columns=columns, sample=data_sample)}")
                chain = validation_prompt | reasoning_llm
                raw_classification = chain.invoke({"columns": columns, "sample": data_sample}).content.strip().lower()
                logger.info(f"LLM Raw Output (Table Target): {raw_classification}")

                target_table = None
                if "outage_reports" in raw_classification:
                    target_table = "outage_reports"
                elif "demand_reports" in raw_classification:
                    target_table = "demand_reports"

                if not target_table:
                    error_msg = f"Data Agent: The uploaded file `{csv_file}` contains irrelevant or unrecognized data and cannot be imported into the system."
                    logger.warning(error_msg)
                    updates["messages"] = state.get("messages", []) + [error_msg]
                else:
                    logger.info(f"Appending data to Table: {target_table}")
                    conn = sqlite3.connect(DB_PATH)
                    df.to_sql(target_table, conn, if_exists="append", index=False)
                    conn.close()

                    msg = f"Data Agent: Analyzed `{csv_file}`. Successfully appended {len(df)} rows to the `{target_table}` database."
                    updates["messages"] = state.get("messages", []) + [msg]
                    chroma_client.delete_collection("llm_prompt_cache")
                    cache_collection = chroma_client.get_or_create_collection("llm_prompt_cache")
                    logger.info("Cache has been cleared")
            except Exception as e:
                error_msg = f"Data Agent Error: Could not load {csv_file}. {e}"
                logger.error(error_msg)
                updates["messages"] = state.get("messages", []) + [error_msg]
        else:
            logger.warning("Intent was 'data_add' but no .csv file was found in the prompt.")
            updates["messages"] = state.get("messages", []) + [
                "Data Agent: Please provide a valid .csv filename to import."]

        return Command(goto="report_agent", update=updates)

    elif intent in ["query", "data_modify", "report"]:
        if intent in ["data_modify"]:
            chroma_client.delete_collection("llm_prompt_cache")
            cache_collection = chroma_client.get_or_create_collection("llm_prompt_cache")
            logger.info("Cache has been cleared")
        current_schema = get_dynamic_schema(DB_PATH)
        action_type = "read-only SELECT query" if intent in ["query",
                                                             "report"] else "Data Manipulation Language (UPDATE/DELETE) query"
        target_source = state.get("source_classification", {}).get("source", "both")
        max_retries = 3
        attempt = 0
        success = False
        error_history = ""
        updates["messages"] = state.get("messages", [])
        while attempt < max_retries and not success:
            sql_prompt = PromptTemplate.from_template("""
                            You are a Principal Data Engineer managing a SQLite database for a major energy grid.
                            Your task is to translate a human user's request into a precise, flawless {action_type}.
                            CURRENT DATABASE SCHEMA:
                            {schema}

                            TARGET DATA SOURCE CLASSIFICATION:
                            The user's intent was classified to primarily target: '{target_source}'
                            If this is 'outage_reports' or 'demand_reports', you MUST heavily prioritize querying that specific table. If 'both', you may join or query across tables if necessary.

                            CRITICAL SCHEMA RULES (ANTI-HALLUCINATION & ANTI-BLEED):
                            1. NEVER invent, guess, or assume table or column names. 
                            2. You MUST ONLY use the exact tables and columns explicitly listed in the CURRENT DATABASE SCHEMA above.
                            3. STRICT TABLE-COLUMN MATCHING: You must verify that the columns you use actually belong to the table you are querying in the FROM clause. Do not use a column from Table A in a query for Table B.
                            4. If the user asks for a category (e.g., "region") that does not exist in the target table, look for the closest geographical equivalent that DOES exist in that specific table (e.g., use "state" or "county" instead).
                            VOCABULARY MAPPING INSTRUCTIONS (Human to SQL):
                            - Apply these conceptual mappings ONLY IF the target columns actually exist in the specific table you are querying:
                            - "blackout" / "power cut" -> look for columns related to `outages`, `status`, or `faults`.
                            - "usage" / "load" / "power draw" -> look for `demand`, `consumption`, or `mw` columns.
                            - "location" / "substation" / "plant" / "region" -> look for `state`, `county`, `location_id`, `facility`, or `site` columns.
                            - Use `LOWER(column_name) LIKE '%term%'` for text searches to handle case-insensitivity.

                            SQLITE DIALECT RULES:
                            - SQLite does not have advanced date types. Use `date(column)`, `datetime(column)`, or `strftime()` for time filtering.
                            - Always use double quotes for column names if they contain spaces or special characters.
                            - CRITICAL: Provide ONLY ONE single SQL statement for SELECT queries. Do NOT separate multiple SELECT queries with semicolons. Combine them using JOINs, UNIONs, aggregations, or return raw data.
                            - If this is a SELECT query, add a `LIMIT 500` at the end to prevent returning excessively large datasets, unless the user explicitly asks for a count or aggregation (SUM, AVG).

                            User Request: {question}
                            {error_context}

                            Return ONLY the raw SQL query. Do not include markdown blocks, explanations, or comments.
                            SQL Query:
                            """)

            error_context = f"\nPREVIOUS ERRORS (Fix these in your new query):\n{error_history}" if error_history else ""
            formatted_sql_prompt = sql_prompt.format(
                action_type=action_type,
                schema=current_schema,
                target_source=target_source,
                question=user_request,
                error_context=error_context
            )
            logger.info(f"LLM Prompt (SQL Generation - Attempt {attempt + 1}):\n{formatted_sql_prompt}")
            chain = sql_prompt | coder_llm
            raw_sql_response = chain.invoke({
                "action_type": action_type,
                "schema": current_schema,
                "target_source": target_source,
                "question": user_request,
                "error_context": error_context
            }).content.strip()
            logger.info(f"LLM Raw Output (SQL - Attempt {attempt + 1}): {raw_sql_response}")

            sql_query = re.sub(r"```[sS][qQ][lL]?", "", raw_sql_response).replace("```", "").strip()
            logger.info(f"Sanitized SQL Query to execute (Attempt {attempt + 1}): {sql_query}")
            try:
                conn = sqlite3.connect(DB_PATH)
                if intent in ["query", "report"]:
                    logger.info("Executing SELECT query...")
                    results_df = pd.read_sql_query(sql_query, conn)
                    logger.info(f"Query returned {len(results_df)} rows.")
                    results_df = results_df.astype(str)
                    updates["db_search_results"] = results_df.to_dict(orient='records')
                    updates["messages"].append(f"Data Agent generated SQL: {sql_query}")
                    next_node = "analysis_agent"
                    success = True
                else:
                    logger.info("Executing DML query...")
                    cursor = conn.cursor()
                    cursor.executescript(sql_query)
                    conn.commit()
                    logger.info("DML executed. Database updated successfully.")
                    updates["messages"].append(f"Data Agent generated SQL: {sql_query}")
                    updates["messages"].append("Data Agent: Database updated successfully.")
                    next_node = "report_agent"
                    success = True
                conn.close()
            except Exception as e:
                attempt += 1
                error_msg = str(e)
                logger.error(f"SQL Execution Failed on attempt {attempt}: {error_msg}")
                error_history += f"Attempt {attempt} query: {sql_query}\nError: {error_msg}\n"
                if attempt >= max_retries:
                    logger.error(f"Data Agent Error: SQL Execution Failed after {max_retries} attempts.")
                    updates["messages"].append(
                        f"Data Agent Error: SQL Execution Failed after {max_retries} attempts. Last error: {error_msg}")
                    next_node = "report_agent"
        logger.info(f"Routing to: {next_node}")
        return Command(goto=next_node, update=updates)

    logger.warning("Fell through data_agent logic. Routing to report_agent by default.")
    return Command(goto="report_agent", update=updates)


def analysis_agent(state: AgentActionState) -> Command[Literal["report_agent"]]:
    logger.info("=== NODE: analysis_agent ===")

    db_results = state.get('db_search_results', 'None')

    # 1. Format the raw list of dicts into an LLM-friendly Markdown table
    if isinstance(db_results, list) and len(db_results) > 0:
        df = pd.DataFrame(db_results)
        # Markdown tables consume far fewer tokens and are easier for the LLM to read than JSON
        formatted_data = df.to_markdown(index=False)
    else:
        formatted_data = str(db_results)

    # Truncate for the console logs to prevent flooding
    db_results_log = formatted_data[:1000] + "... [TRUNCATED]" if len(formatted_data) > 1000 else formatted_data

    analysis_prompt = f"""
    You are an Expert Energy Data Analyst. Your job is to interpret database returns and unstructured logs, translating them into meaningful insights.

    User Query: {state['user_request']}

    Database Results: 
    {formatted_data}

    Unstructured Data Logs: {state.get('issue_search_results', 'None')}

    INSTRUCTIONS:
    1. If the database results are empty or "None", state explicitly: "No data was found matching the parameters of the query."
    2. If the data is a large list (e.g., grouped by state/region), summarize the top 10 most significant rows (highest values) in a clean markdown table.
    3. Identify any obvious anomalies or trends (e.g., "California had significantly higher outages than other regions").
    4. Keep your analysis objective, data-driven, and focused strictly on answering the User Query.
    5. CRITICAL: You must provide a written analysis. Do not return an empty response.

    Draft your analytical findings below:
    """

    logger.info(f"LLM Prompt (Analysis):\n[Prompt includes DB Results: {db_results_log}]\n...")
    response = reasoning_llm.invoke(analysis_prompt)

    analysis_content = response.content.strip()

    # 2. FALLBACK MECHANISM: If the LLM chokes and returns empty, pass the data through manually
    if not analysis_content:
        logger.warning("Analysis Agent returned an empty string. Falling back to passing raw formatted data.")
        analysis_content = f"The LLM failed to summarize the data. Here is the raw data return:\n{formatted_data}"

    logger.info(f"LLM Output (Analysis):\n{analysis_content}")

    messages = state.get("messages", []) + [f"Analysis Agent: {analysis_content}"]
    return Command(goto="report_agent", update={"messages": messages})


def report_agent(state: AgentActionState) -> Command[Literal["draft_response"]]:
    logger.info("=== NODE: report_agent ===")

    report_prompt = f"""
    You are the Lead Communications Officer for a Power Grid Authority. 
    Take the raw agent logs and data analysis provided below and draft a clear, professional, and concise response to the user.

    User's Original Request: {state['user_request']}

    Agent Logs & Data Analysis:
    {state.get('messages', ['No analysis available.'])}

    FORMATTING RULES:
    - If the user asked a simple query, provide a direct, concise answer.
    - If the user asked for a report, format the response with professional headers (e.g., **Executive Summary**, **Key Metrics**, **Details**).
    - Exclude internal system jargon (e.g., do not say "The SQL query returned...", instead say "The grid data indicates...").
    - Ensure the tone is authoritative, reassuring, and clear.
    """

    logger.info(f"LLM Prompt (Report Generation):\n{report_prompt}")
    response = reasoning_llm.invoke(report_prompt)
    logger.info(f"LLM Output (Report Draft):\n{response.content}")

    return Command(goto="draft_response", update={"drafted_response": response.content})


def draft_response(state: AgentActionState):
    logger.info("=== NODE: draft_response ===")

    if state.get("classification", {}).get("intent") == "irrelevant":
        logger.info("Intent was irrelevant. Bypassing drafted response for default fallback.")
        return {
            "drafted_response": "I am an energy management assistant. I can only assist with power supply, grid demand, outage reporting, and related data queries. How can I help you with your energy infrastructure today?"}

    logger.info("Final response ready for delivery.")
    return {"drafted_response": state["drafted_response"]}
