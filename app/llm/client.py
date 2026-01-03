from functools import lru_cache

from langchain_ollama import ChatOllama

from app.core import OLLAMA_MODEL, get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_chat_model() -> ChatOllama:
    """
    Единая точка создания LLM-клиента.
    """
    model = OLLAMA_MODEL

    metricks = {
        "provider": "ollama",
        "model": model
    }
    logger.info(f"Using LLM model: {metricks}")

    return ChatOllama(
        model=model,
    )