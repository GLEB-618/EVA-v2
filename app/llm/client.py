from functools import lru_cache

from langchain_ollama import ChatOllama

from app.core import OLLAMA_MODEL, get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_chat_model(temperature: float = 0.0) -> ChatOllama:
    """
    Единая точка создания LLM-клиента.
    Меняешь модель/температуру тут — и весь проект подхватывает.
    """
    model = OLLAMA_MODEL

    return ChatOllama(
        model=model,
        temperature=temperature,
    )

def describe_llm() -> dict:
    """
    Просто чтобы логировать, что за модель/настройки используются.
    """
    llm = get_chat_model()
    metricks = {
        "provider": "ollama",
        "base_url": getattr(llm, "base_url", None),
        "model": getattr(llm, "model", None),
        "temperature": getattr(llm, "temperature", None),
    }
    logger.info(f"Using LLM model: {metricks}")
    return metricks