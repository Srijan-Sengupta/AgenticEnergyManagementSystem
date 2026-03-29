import re
import sqlite3
import pandas as pd
import logging
from typing import Literal
from langchain_core.prompts import PromptTemplate
from langgraph.types import Command

from app.core.state import AgentActionState, ActionClassification, SourceClassification
from app.core.config import DB_PATH
from app.db.database import get_dynamic_schema
from llm import reasoning_llm, coder_llm

logger = logging.getLogger(__name__)


def classify_intent(state: AgentActionState) -> Command[Literal["classify_sources", "draft_response"]]:
    logger.info("--- NODE: classify_intent ---")
    structured_text_llm = reasoning_llm.with_structured_output(ActionClassification)

    classification_prompt = f"""
        You are an expert energy supply analysis and report generation agent.
        Analyze the user's message and classify its intent.

        CRITICAL RULES:
        1. If the user asks to import, load, update, or add data (e.g., from a CSV), classify it as 'data_add' or 'data_modify'.
        2. If the user asks about power grids, energy demand, or outages, classify it as 'query' or 'report'.
        3. ONLY if the request is completely unrelated to energy systems or data management, classify it as 'irrelevant'.

        User's request: {state['user_request']}
        """

    classification = structured_text_llm.invoke(classification_prompt)
    logger.info(f"Classified Intent: {classification.get('intent', 'Unknown')}")

    goto = "draft_response" if classification["intent"] == "irrelevant" else "classify_sources"
    return Command(goto=goto, update={"classification": classification})


def classify_sources(state: AgentActionState) -> Command[Literal["data_agent", "report_agent", "analysis_agent"]]:
    logger.info("--- NODE: classify_sources ---")
    structured_text_llm = reasoning_llm.with_structured_output(SourceClassification)

    classifications_prompt = f"""
    Which sources contain the answer to the user's query?
    - "outage_reports": Unstructured text about grid outages/faults.
    - "demand_reports": Structured logs of consumption/generation.
    - "both": If both are needed.
    User's query: {state['user_request']}
    Intent Summary: {state["classification"].get("summary", "Unknown")}
    """

    classification = structured_text_llm.invoke(classifications_prompt)
    intent = state["classification"]["intent"]

    if intent in ["data_add", "data_modify", "query", "report"]:
        goto = "data_agent"
    else:
        goto = "report_agent"

    return Command(goto=goto, update={"source_classification": classification})


def data_agent(state: AgentActionState) -> Command[Literal["analysis_agent", "report_agent"]]:
    logger.info("--- NODE: data_agent ---")
    intent = state["classification"]["intent"]
    user_request = state["user_request"]
    updates = {}

    if intent == "data_add":
        match = re.search(r'([\w\-./\\]+\.csv)', user_request)
        if match:
            csv_file = match.group(1)
            try:
                df = pd.read_csv(csv_file)
                columns = ", ".join(df.columns)
                data_sample = df.head(1).to_markdown(index=False)

                naming_prompt = PromptTemplate.from_template("""
                    You are a database architect. Look at the following sample of a CSV file.
                    Columns: {columns}
                    Data Sample:
                    {sample}
                    Suggest a single, descriptive database table name using lowercase letters and underscores.
                    CRITICAL: Output ONLY the table name.
                    """)

                chain = naming_prompt | reasoning_llm
                raw_name = chain.invoke({"columns": columns, "sample": data_sample}).content.strip()
                table_name = re.sub(r'[^a-z0-9_]', '', raw_name.lower()) or "unnamed_dataset"

                conn = sqlite3.connect(DB_PATH)
                df.to_sql(table_name, conn, if_exists="replace", index=False)
                conn.close()

                msg = f"Data Agent: Analyzed `{csv_file}`. Automatically created table: `{table_name}` ({len(df)} rows)."
                updates["messages"] = state.get("messages", []) + [msg]
            except Exception as e:
                updates["messages"] = state.get("messages", []) + [f"Data Agent Error: Could not load {csv_file}. {e}"]
        else:
            updates["messages"] = state.get("messages", []) + [
                "Data Agent: Please provide a valid .csv filename to import."]
        return Command(goto="report_agent", update=updates)

    elif intent in ["query", "data_modify", "report"]:
        current_schema = get_dynamic_schema(DB_PATH)
        action_type = "read-only SELECT query" if intent in ["query", "report"] else "Data Manipulation Language query"

        sql_prompt = PromptTemplate.from_template("""
                    You are an expert SQLite developer. Write a {action_type} to fulfill the user's request.
                    Only return the raw SQL query.
                    Current Database Schema: {schema}
                    User Request: {question}
                    SQL Query:
                    """)

        chain = sql_prompt | coder_llm
        raw_sql_response = chain.invoke(
            {"action_type": action_type, "schema": current_schema, "question": user_request}).content.strip()
        sql_query = re.sub(r"```[sS][qQ][lL]?", "", raw_sql_response).replace("```", "").strip()

        updates["messages"] = state.get("messages", []) + [f"Data Agent generated SQL: {sql_query}"]

        try:
            conn = sqlite3.connect(DB_PATH)
            if intent in ["query", "report"]:
                results_df = pd.read_sql_query(sql_query, conn)
                updates["db_search_results"] = results_df.to_dict(orient='records')
                next_node = "analysis_agent"
            else:
                cursor = conn.cursor()
                cursor.execute(sql_query)
                conn.commit()
                updates["messages"].append(
                    f"Data Agent: Database updated successfully. {cursor.rowcount} row(s) affected.")
                next_node = "report_agent"
            conn.close()
        except Exception as e:
            updates["messages"].append(f"Data Agent Error: SQL Execution Failed: {str(e)}")
            next_node = "report_agent"

        return Command(goto=next_node, update=updates)

    return Command(goto="report_agent", update=updates)


def analysis_agent(state: AgentActionState) -> Command[Literal["report_agent"]]:
    logger.info("--- NODE: analysis_agent ---")
    analysis_prompt = f"""
    Analyze the following energy data to answer the user's query.
    User Query: {state['user_request']}
    Database Results: {state.get('db_search_results', 'None')}
    Unstructured Data: {state.get('issue_search_results', 'None')}
    """
    response = reasoning_llm.invoke(analysis_prompt)
    messages = state.get("messages", []) + [f"Analysis Agent: {response.content}"]
    return Command(goto="report_agent", update={"messages": messages})


def report_agent(state: AgentActionState) -> Command[Literal["draft_response"]]:
    logger.info("--- NODE: report_agent ---")
    report_prompt = f"""
    Draft a concise, professional response based on the agent logs and analysis below. 
    Agent Logs & Analysis:
    {state.get('messages', ['No analysis available.'])}
    """
    response = reasoning_llm.invoke(report_prompt)
    return Command(goto="draft_response", update={"drafted_response": response.content})


def draft_response(state: AgentActionState):
    logger.info("--- NODE: draft_response ---")
    if state.get("classification", {}).get("intent") == "irrelevant":
        return {
            "drafted_response": "I am an energy management assistant. I can only assist with power supply, demand, and outage queries."}
    return {"drafted_response": state["drafted_response"]}