"""Единственное место, где создаются модели. Тесты подменяют get_chat_model и get_embeddings на фейки."""
from functools import lru_cache
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from ai.usage import UsageCallback
from config import LLM_MODEL, EMBEDDING_MODEL


@lru_cache
def get_chat_model(temperature: float = 0.0) -> BaseChatModel:
    # У Ollama по умолчанию маленькое окно контекста, а в промпт попадают куски лекций. num_predict - страховка:
    # без него модель изредка зацикливается внутри JSON и генерирует, пока не забьёт всё окно
    extra = {'num_ctx': 8192, 'num_predict': 1536} if LLM_MODEL.startswith('ollama:') else {}
    return init_chat_model(LLM_MODEL, temperature=temperature, **extra)


@lru_cache
def get_embeddings() -> Embeddings:
    return init_embeddings(EMBEDDING_MODEL)


def run_config(feature: str, **metadata) -> RunnableConfig:
    """run_name, tags и metadata попадают в трейсы, а UsageCallback пишет токены и задержку в ai_usage."""
    return {'run_name': feature, 'tags': ['ustaz', feature], 'metadata': metadata,
            'callbacks': [UsageCallback(feature)]}
