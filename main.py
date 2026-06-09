import time

import streamlit as st
import os
import sqlite3
import pandas as pd

from app.agent.graph import app as agent_app
from app.core.config import DB_PATH, DATA_DIR

def stream_response(text):
    """Generator to simulate streaming text output."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)

st.set_page_config(
    page_title="Energy Management Assistant",
    layout="wide"
)

st.title("Agent-Based Energy Management Assistant")

# --- UI Layout: Sidebar ---
with st.sidebar:
    st.header("Data Management")
    st.write("Upload CSVs.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], max_upload_size=4096)

    if uploaded_file is not None:
        file_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Saved `{file_path}` locally.")

        if st.button("Import Data to Database"):
            with st.spinner(f"Instructing Data Agent to import {file_path}..."):
                initial_state = {
                    "user_request": f"Import the data from {file_path}",
                    "messages": []
                }
                try:
                    result = agent_app.invoke(initial_state)
                    st.info(result["drafted_response"])
                except Exception as e:
                    st.error(f"Import failed: {e}")

    st.markdown("---")
    st.subheader("Sample Queries:")
    st.code('"Summarize outage_reports by region"')
    st.code('"What is the total demand in the demand_reports table?"')

# --- UI Layout: Main Tabs ---
tab1, tab2 = st.tabs(["Assistant Chat", "Database Viewer"])

# --- TAB 1: Chat Interface ---
with tab1:
    chat_container = st.container(height=600)

    if "messages" not in st.session_state:
        st.session_state.messages = []

        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    # Persist the logs/thoughts in an expander for past messages
                    if message.get("logs"):
                        with st.expander("View Agent Internal Execution Logs"):
                            for log in message["logs"]:
                                st.info(log)

    if prompt := st.chat_input("Ask about energy demand, supply, or outages..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                agent_logs = []
                final_state = None

                try:
                    initial_state = {
                        "user_request": prompt,
                        "messages": []
                    }

                    with st.status("Initializing agents...", expanded=True) as status:
                        for event in agent_app.stream(initial_state):
                            for node_name, node_state in event.items():
                                status.update(label=f"Agent `{node_name}` is working...", state="running")
                                final_state = node_state

                                st.write(f"Completed step: **{node_name}**")

                        status.update(label="Analysis complete!", state="complete", expanded=False)

                    if final_state:
                        raw_response = final_state.get("drafted_response", "Error: No response drafted.")

                        if "<think>" in raw_response:
                            parts = raw_response.split("</think>")
                            think_content = parts[0].replace("<think>", "").strip()
                            response = parts[-1].strip()
                            if think_content:
                                agent_logs.append(f"DeepSeek Reasoning:\n{think_content}")
                        else:
                            response = raw_response

                        if final_state.get("messages"):
                            agent_logs.extend(final_state["messages"])
                    else:
                        response = "Error: Graph execution failed to return a state."

                except Exception as e:
                    response = f"An error occurred during execution: {e}"
                    st.error(response)

                if "response" in locals():
                    st.write_stream(stream_response(response))

                if agent_logs:
                    with st.expander("View Agent Internal Execution Logs"):
                        for log in agent_logs:
                            st.info(log)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response if "response" in locals() else "Error",
                    "logs": agent_logs
                })
# --- TAB 2: Database Viewer ---
with tab2:
    st.header("Database Tables")
    db_path = DB_PATH

    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        if not tables:
            st.info("No tables found in the database. Upload a CSV in the sidebar to get started.")
        else:
            selected_table = st.selectbox("Select a table to view:", tables)

            try:
                df = pd.read_sql_query(f"SELECT * FROM {selected_table} LIMIT 1000", conn)
                st.dataframe(df, width='stretch')
                st.caption(f"Showing {len(df)} rows from `{selected_table}`.")
            except Exception as e:
                st.error(f"Could not load table: {e}")

        conn.close()
    else:
        st.warning("Database has not been created yet. Upload and import a file first.")
