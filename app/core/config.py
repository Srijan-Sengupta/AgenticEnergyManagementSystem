import logging
import os

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
DATA_DIR = os.path.join(os.getcwd(), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Database configurations
DB_PATH = os.getenv("ENERGY_DB_PATH", "data/energy_data.db")

# LLM Configurations
REASONING_MODEL = os.getenv("REASONING_MODEL", "deepseek-r1:7b")
CODER_MODEL = os.getenv("CODER_MODEL", "qwen2.5-coder:latest") # or codeqwen:latest
EMBEDDING_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
SEMANTIC_CACHE=os.getenv("SEMANTIC_CACHE", "data/semantic_cache")