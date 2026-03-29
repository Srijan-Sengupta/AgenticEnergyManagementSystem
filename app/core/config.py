import os

# Database configurations
DB_PATH = os.getenv("ENERGY_DB_PATH", "../data/energy_data.db")

# LLM Configurations
REASONING_MODEL = os.getenv("REASONING_MODEL", "deepseek-r1:7b")
CODER_MODEL = os.getenv("CODER_MODEL", "codeqwen:latest")