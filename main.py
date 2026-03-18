import streamlit as st
import os

# Import the compiled LangGraph app from your backend file
# (Ensure your previous code is saved in a file named agent_backend.py)
from agent_backend import app as agent_app

# --- UI Configuration ---
st.set_page_config(
    page_title="Energy Management Assistant",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Agent-Based Energy Management Assistant")
st.markdown("""
This system uses multi-agent orchestration to retrieve and analyze power grid data. 
You can upload CSV datasets and ask questions in natural language.
""")

# --- Sidebar: Data Upload ---
with st.sidebar:
    st.header("📂 Data Management")
    st.write("Upload your demand or outage reports here.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        # Save the uploaded file to the local directory so the data_agent can access it
        file_path = uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Saved `{file_path}` locally.")

        # Provide a quick-action button to trigger the import via the agent
        if st.button("Import Data to Database"):
            with st.spinner("Instructing Data Agent to import..."):
                initial_state = {
                    "user_request": f"Import the data from {file_path}",
                    "messages": []
                }
                result = agent_app.invoke(initial_state)
                st.info(result["drafted_response"])

    st.markdown("---")
    st.subheader("Sample Queries:")
    st.code('"What was peak demand?"\n"Update the demand for 2025-08-02 to 1500"\n"Summarize outages by region"')

# --- Main Chat Interface ---

# Initialize chat history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("Ask about energy demand, supply, or outages..."):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process the request through LangGraph
    with st.chat_message("assistant"):
        with st.spinner("Agents are analyzing..."):
            try:
                # Prepare the initial state for LangGraph
                initial_state = {
                    "user_request": prompt,
                    "messages": []
                }

                # Invoke the LangGraph orchestration
                result = agent_app.invoke(initial_state)

                # Extract the final drafted response
                response = result.get("drafted_response", "Error: No response drafted.")

                # Display the response
                st.markdown(response)

                # Expandable section to show the agent's internal thought process/steps
                if "messages" in result and result["messages"]:
                    with st.expander("View Agent Internal Execution Logs"):
                        for log in result["messages"]:
                            st.text(log)

            except Exception as e:
                response = f"An error occurred during execution: {e}"
                st.error(response)

        # Save assistant response to UI history
        st.session_state.messages.append({"role": "assistant", "content": response})