import logging

import chromadb
from types import MethodType
from langchain_core.messages import AIMessage
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import SEMANTIC_CACHE
logger = logging.getLogger(__name__)
embedding_model = HuggingFaceEmbeddings(
	model_name="BAAI/bge-small-en-v1.5",
	model_kwargs={'device': 'cpu'},
	encode_kwargs={'normalize_embeddings': True}
)

chroma_client = chromadb.PersistentClient(path=SEMANTIC_CACHE)
cache_collection = chroma_client.get_or_create_collection(name="llm_prompt_cache")


def wrap_with_cache(llm_instance, threshold=0.98):
	original_invoke = llm_instance.invoke

	def cached_invoke(self, prompt_value, config=None, **kwargs):
		try:
			prompt_string = prompt_value.to_string()
		except AttributeError:
			prompt_string = str(prompt_value)

		query_vector = embedding_model.embed_query(prompt_string)
		cache_res = cache_collection.query(query_embeddings=[query_vector], n_results=1)

		if cache_res and cache_res['distances'] and cache_res['distances'][0]:
			distance = cache_res['distances'][0][0]
			if distance < (1 - threshold):
				cached_text = cache_res['metadatas'][0][0]['response_text']
				# SAFETY: Don't return an empty string if we accidentally cached one earlier
				logger.info("Cache hit! %s", cached_text)
				if cached_text.strip():
					return AIMessage(content=cached_text)

		response = original_invoke(prompt_value, config=config, **kwargs)

		# FIX: Do not cache Structured Outputs (tool calls) or empty responses
		if getattr(response, 'tool_calls', None) or not response.content.strip():
			return response

		cache_collection.add(
			embeddings=[query_vector],
			metadatas=[{"response_text": response.content}],
			ids=[prompt_string[:500]]
		)
		return response

	# Bypass Pydantic's strict attribute locking
	object.__setattr__(llm_instance, 'invoke', MethodType(cached_invoke, llm_instance))

	return llm_instance