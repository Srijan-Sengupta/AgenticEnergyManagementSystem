# Agent-Based Energy Demand and Supply Management Assistant

This repository contains the application code and system design for the Agentic Energy Management System. The system is designed to retrieve and analyze structured consumption logs alongside unstructured outage reports to provide natural language decision support for the power sector.

## Problem Statement

The objective of this project is to orchestrate an Agentic AI system that processes diverse energy datasets without requiring model fine-tuning. It leverages a multi-agent workflow consisting of a Data Agent, Analysis Agent, and Report Agent to answer user queries and provide actionable insights.

## Technical Stack

* **LLM:** Hugging Face
* **Agent Orchestration:** LangChain
* **Vector Database:** ChromaDB (for embedding and storing outage reports)
* **Data Processing:** Pandas
* **Visualization:** Matplotlib
* **User Interface:** Streamlit

## Project Structure

* **`app/agent/`**: Contains the core logic for the AI agents, including the execution graphs (`graph.py`), language model integrations (`llm.py`), and agent steps (`nodes.py`)
* **`app/core/`**: Manages the application's internal state, environment configuration, and memory caching via `state.py`, `config.py`, and `cache.py`
* **`app/db/`**: Handles data retrieval and database interactions through `database.py`
* **`main.py` & `cli.py`**: The primary entry points for running the main application server and the command-line interface.
* **Jupyter Notebooks**: Includes `System Design.ipynb`, `Main Code Logic.ipynb`, `ColabNotebook.ipynb`, and `Tesing.ipynb` for architecture documentation, data loading tests, and workflow orchestration.
* **Configuration Files**: Includes `requirements.txt` for dependencies, `.env.example` for environment variables, and `.steamlit/config.toml` for the UI configuration.

## Installation

1. Clone the repository and navigate to the project directory.
2. Create and activate a Python virtual environment.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
