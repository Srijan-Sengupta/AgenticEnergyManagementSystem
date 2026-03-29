import os
import re
import sqlite3
import pandas as pd
import logging
from typing import TypedDict, Literal, Optional, List, Dict, Any

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langgraph.types import Command
from langgraph.constants import START, END
from langgraph.graph import StateGraph

# --- Setup Debug Logging ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Type Definitions ---
class ActionClassification(TypedDict):
    intent: Literal["data_add", "data_modify", "query", "report", "irrelevant"]
    summary: str
    reasoning: str


class SourceClassification(TypedDict):
    source: Literal["outage_reports", "demand_reports", "both"]
    reasoning: str


class AgentActionState(TypedDict):
    user_request: str
    classification: Optional[ActionClassification]
    source_classification: Optional[SourceClassification]

    issue_search_results: Optional[List[str]]
    # Renamed from demand_search_results to db_search_results to handle ALL tables
    db_search_results: Optional[Dict[str, Any]]

    messages: List[str]
    drafted_response: Optional[str]

# --- LLM Setup ---
reasoning_llm = ChatOllama(
    model="deepseek-r1:7b",
    reasoning=True,
    temperature=0.5,
    num_ctx=8192
)

coder_llm = ChatOllama(
    model="qwen2.5-coder:latest",
    temperature=0.0,
    num_ctx=2048
)


# --- Helper Functions ---
def setup_database(csv_path: str, db_path: str = "energy_data.db"):
    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    conn = sqlite3.connect(db_path)
    df.to_sql('demand_reports', conn, if_exists='replace', index=False)
    schema = pd.io.sql.get_schema(df, 'demand_reports')
    conn.close()
    return schema


def get_dynamic_schema(db_path: str = "energy_data.db") -> str:
    try:
        if not os.path.exists(db_path):
            return "No tables exist in the database yet."
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
        schemas = [row[0] for row in cursor.fetchall() if row[0] is not None]
        conn.close()
        return "\n\n".join(schemas) if schemas else "No tables exist in the database yet."
    except Exception as e:
        return f"Error reading schema: {e}"


# --- Agent Nodes ---
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
    logger.info(f"Classified Source: {classification.get('source', 'Unknown')} | Reasoning: {classification.get('reasoning', 'None')}")

    intent = state["classification"]["intent"]

    # BUG FIX: Ensure "report" also goes to data_agent to execute SQL data extraction
    if intent in ["data_add", "data_modify", "query", "report"]:
        goto = "data_agent"
    else:
        goto = "report_agent"

    logger.info(f"Routing to: {goto}")
    return Command(goto=goto, update={"source_classification": classification})


def data_agent(state: AgentActionState) -> Command[Literal["analysis_agent", "report_agent"]]:
    logger.info("--- NODE: data_agent ---")
    intent = state["classification"]["intent"]
    user_request = state["user_request"]
    updates = {}
    db_path = "energy_data.db"

    # --- DYNAMIC IMPORT ---
    # ---------------------------------------------------------
    # ACTION: IMPORT & AUTO-PROFILE DATA
    # ---------------------------------------------------------
    if intent == "data_add":
        # Extract the file path from the user request
        match = re.search(r'([\w\-./\\]+\.csv)', user_request)
        if match:
            csv_file = match.group(1)
            try:
                # 1. Load the data
                df = pd.read_csv(csv_file)

                # 2. Extract a sample to show the LLM
                columns = ", ".join(df.columns)
                data_sample = df.head(1).to_markdown(index=False)

                # 3. Ask the LLM to analyze the data and name the table
                naming_prompt = PromptTemplate.from_template("""
                    You are a database architect. Look at the following sample of a CSV file.

                    Columns: {columns}
                    Data Sample:
                    {sample}

                    Based on the semantic meaning of this data, suggest a single, descriptive database table name using lowercase letters and underscores (e.g., 'power_outages', 'daily_grid_demand', 'sensor_readings').
                    CRITICAL: Output ONLY the table name. No markdown, no quotes, no explanations.
                    """)

                chain = naming_prompt | reasoning_llm
                raw_name = chain.invoke({
                    "columns": columns,
                    "sample": data_sample
                }).content.strip()

                # 4. Clean the LLM output to ensure it's a valid SQL table name
                table_name = re.sub(r'[^a-z0-9_]', '', raw_name.lower())
                if not table_name:
                    table_name = "unnamed_dataset"  # Fallback safeguard

                # 5. Save to SQLite using the intelligent name
                conn = sqlite3.connect(db_path)
                df.to_sql(table_name, conn, if_exists="replace", index=False)
                conn.close()

                msg = f"Data Agent: Analyzed `{csv_file}`. Detected contents and automatically created table: `{table_name}` ({len(df)} rows)."
                logger.info(msg)
                updates["messages"] = state.get("messages", []) + [msg]

            except Exception as e:
                logger.error(f"Import Failed: {e}")
                updates["messages"] = state.get("messages", []) + [f"Data Agent Error: Could not load {csv_file}. {e}"]
        else:
            updates["messages"] = state.get("messages", []) + [
                "Data Agent: Please provide a valid .csv filename to import."]

        return Command(goto="report_agent", update=updates)
    # --- CRUD OPERATIONS ---
    elif intent in ["query", "data_modify", "report"]:
        current_schema = get_dynamic_schema(db_path)
        action_type = "read-only SELECT query" if intent in ["query",
                                                             "report"] else "Data Manipulation Language (UPDATE, DELETE, INSERT) query"
        logger.info("current schema: {}".format(current_schema))
        sql_prompt = PromptTemplate.from_template("""
                    You are an expert SQLite developer. Write a {action_type} to fulfill the user's request.
                    Only return the raw SQL query. Do not wrap it in markdown formatting like ```sql.

                    Current Database Schema:
                    {schema}

                    CRITICAL INSTRUCTIONS:
                    1. Intelligently map the user's vocabulary to the actual column names in the schema (e.g., if they ask for 'state' but the schema has 'region', use 'region').
                    2. If a requested concept does not match the schema exactly, write the closest possible valid query using the available columns.

                    User Request: {question}
                    SQL Query:
                    """)

        chain = sql_prompt | coder_llm
        raw_sql_response = chain.invoke({
            "action_type": action_type,
            "schema": current_schema,
            "question": user_request
        }).content.strip()

        # Clean markdown if present
        sql_query = re.sub(r"```[sS][qQ][lL]?", "", raw_sql_response).replace("```", "").strip()
        logger.info(f"Generated Clean SQL: {sql_query}")

        # Add generated SQL to the state messages so it appears in Streamlit UI logs
        updates["messages"] = state.get("messages", []) + [f"Data Agent generated SQL: {sql_query}"]

        try:
            conn = sqlite3.connect(db_path)
            if intent in ["query", "report"]:
                results_df = pd.read_sql_query(sql_query, conn)
                logger.info(
                    f"Query returned {len(results_df)} rows. Sample Data: {results_df.to_dict(orient='records')}")

                # Save to the new generalized state key
                updates["db_search_results"] = results_df.to_dict(orient='records')
                #updates["messages"].append(f"Data Agent: Extracted {len(results_df)} rows via SQL.")
                next_node = "analysis_agent"
            else:
                cursor = conn.cursor()
                cursor.execute(sql_query)
                conn.commit()
                msg = f"Data Agent: Database updated successfully. {cursor.rowcount} row(s) affected."
                logger.info(msg)
                updates["messages"].append(msg)
                next_node = "report_agent"
            conn.close()
        except Exception as e:
            error_msg = f"SQL Execution Failed: {str(e)}\nAttempted Query: {sql_query}"
            logger.error(error_msg)
            updates["messages"].append(f"Data Agent Error: {error_msg}")
            next_node = "report_agent"

        return Command(goto=next_node, update=updates)

    return Command(goto="report_agent", update=updates)


def analysis_agent(state: AgentActionState) -> Command[Literal["report_agent"]]:
    logger.info("--- NODE: analysis_agent ---")

    analysis_prompt = f"""
    Analyze the following energy data to answer the user's query.

    CRITICAL RULES:
    1. ONLY extract and analyze the factual data provided.
    2. DO NOT correct grammar, spelling, or formatting of the system logs or previous messages.
    3. DO NOT comment on how the data was retrieved. Just provide the mathematical or factual answer.

    User Query: {state['user_request']}

    Database Results (Structured SQL Output): {state.get('db_search_results', 'None')}
    Unstructured Data: {state.get('issue_search_results', 'None')}
    """
    response = reasoning_llm.invoke(analysis_prompt)

    # DEBUG LOG ADDED HERE
    logger.info(f"Analysis Agent Output:\n{response.content}\n{'-' * 40}")

    messages = state.get("messages", []) + [f"Analysis Agent: {response.content}"]
    logger.info("Analysis complete.")
    return Command(goto="report_agent", update={"messages": messages})


def report_agent(state: AgentActionState) -> Command[Literal["draft_response"]]:
    logger.info("--- NODE: report_agent ---")

    report_prompt = f"""
    Draft a concise, professional response based on the agent logs and analysis below. 

    CRITICAL RULES:
    1. DO NOT output generic apologies, platitudes, or phrases like "To provide a meaningful analysis, please ensure...".
    2. If the logs show an error (like a SQL failure), tell the user exactly what went wrong in plain English (e.g., "I encountered a SQL error because the column does not exist").
    3. If the analysis is successful, output ONLY the final answer. DO NOT output your internal thought processes or meta-commentary.

    Agent Logs & Analysis:
    {state.get('messages', ['No analysis available.'])}
    """
    response = reasoning_llm.invoke(report_prompt)

    # DEBUG LOG ADDED HERE
    logger.info(f"Report Agent Drafted Response:\n{response.content}\n{'-' * 40}")

    return Command(goto="draft_response", update={"drafted_response": response.content})

def draft_response(state: AgentActionState):
    logger.info("--- NODE: draft_response ---")
    if state.get("classification", {}).get("intent") == "irrelevant":
        return {
            "drafted_response": "I am an energy management assistant. I can only assist with power supply, demand, and outage queries."}
    return {"drafted_response": state["drafted_response"]}


# --- Graph Compilation ---
builder = StateGraph(AgentActionState)
builder.add_node("classify_intent", classify_intent)
builder.add_node("classify_sources", classify_sources)
builder.add_node("data_agent", data_agent)
builder.add_node("analysis_agent", analysis_agent)
builder.add_node("report_agent", report_agent)
builder.add_node("draft_response", draft_response)

builder.add_edge(START, "classify_intent")
builder.add_edge("draft_response", END)
app = builder.compile()


# --- CLI Setup ---
def run_energy_assistant_cli(agent_app):
    # FIX: Removed emojis from the CLI header
    print("=" * 60)
    print(" Agent-Based Energy Management Assistant (CLI Mode) ")
    print("=" * 60)
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            break
        if not user_input: continue

        prompt = user_input
        if user_input.lower().startswith("import "):
            filename = user_input.split(" ", 1)[1].strip()
            prompt = f"Import the data from {filename}"

        initial_state = {"user_request": prompt, "messages": []}
        try:
            result = agent_app.invoke(initial_state)
            print(f"Assistant: {result.get('drafted_response', 'Error')}")
        except Exception as e:
            print(f"\n[Execution Error]: {e}")


# BUG FIX: Prevent this from running automatically when imported by Streamlit
if __name__ == "__main__":
    run_energy_assistant_cli(app)