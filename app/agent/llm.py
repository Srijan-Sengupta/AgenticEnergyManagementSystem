#from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from app.core.config import REASONING_MODEL, CODER_MODEL
import os
import getpass

"""reasoning_llm = ChatOllama(
    model=REASONING_MODEL,
    reasoning=True,
    temperature=0.5,
    num_ctx=8192
)

coder_llm = ChatOllama(
    model=CODER_MODEL,
    temperature=0.0,
    num_ctx=2048
)"""

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

reasoning_llm = ChatGroq(model="qwen/qwen3-32b", reasoning_format="hidden", temperature=0.5)
coder_llm = ChatGroq(model="qwen/qwen3-32b", temperature=0.0, reasoning_format="hidden")
