import streamlit as st
import os
import sqlite3
import pandas as pd

# Import the compiled LangGraph app
from agent_backend import app as agent_app

st.set_page_config(
    page_title="Energy Management Assistant",
    layout="wide"
)

st.title("Agent-Based Energy Management Assistant")

# --- UI Layout: Sidebar ---
with st.sidebar:
    st.header("Data Management")
    st.write("Upload CSVs. The filename becomes the table name (e.g., `outage_reports.csv` -> `outage_reports`).")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        file_path = uploaded_file.name
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
    st.code('"Update the status in outage_reports to Resolved where region is North"')

# --- UI Layout: Main Tabs ---
tab1, tab2 = st.tabs(["Assistant Chat", "Database Viewer"])

# --- TAB 1: Chat Interface ---
# --- TAB 1: Chat Interface ---
with tab1:
    # 1. Create a fixed-height, scrollable container for the chat history
    chat_container = st.container(height=600)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 2. Render all existing messages INSIDE the scrollable container
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # 3. The chat input natively sticks to the bottom of the active tab/screen
    if prompt := st.chat_input("Ask about energy demand, supply, or outages..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 4. Ensure new messages are also written INSIDE the container
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Agents are analyzing (Check terminal for debug logs)..."):
                    try:
                        initial_state = {
                            "user_request": prompt,
                            "messages": []
                        }

                        result = agent_app.invoke(initial_state)
                        response = result.get("drafted_response", "Error: No response drafted.")

                        # Cleanly strip out DeepSeek's <think> tags if they leak into the final response
                        if "<think>" in response:
                            response = response.split("</think>")[-1].strip()

                        st.markdown(response)

                        # Show internal logs clearly
                        if result.get("messages"):
                            with st.expander("View Agent Internal Execution Logs"):
                                for log in result["messages"]:
                                    st.info(log)

                    except Exception as e:
                        response = f"An error occurred during execution: {e}"
                        st.error(response)

                # Save the assistant's response to state
                st.session_state.messages.append({"role": "assistant", "content": response})
# --- TAB 2: Database Viewer ---
with tab2:
    st.header("Database Tables")
    db_path = "energy_data.db"

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
                df = pd.read_sql_query(f"SELECT * FROM {selected_table}", conn)
                # FIX APPLIED HERE: Replaced use_container_width with width='stretch'
                st.dataframe(df, width='stretch')
                st.caption(f"Showing {len(df)} rows from `{selected_table}`.")
            except Exception as e:
                st.error(f"Could not load table: {e}")

        conn.close()
    else:
        st.warning("Database has not been created yet. Upload and import a file first.")