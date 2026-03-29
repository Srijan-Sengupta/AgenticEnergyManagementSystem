from langchain_ollama import ChatOllama
from app.core.config import REASONING_MODEL, CODER_MODEL

reasoning_llm = ChatOllama(
    model=REASONING_MODEL,
    reasoning=True,
    temperature=0.5,
    num_ctx=8192
)

coder_llm = ChatOllama(
    model=CODER_MODEL,
    temperature=0.0,
    num_ctx=2048
)