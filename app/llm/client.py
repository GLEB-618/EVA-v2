from functools import lru_cache
from typing import Literal
from langchain_ollama import ChatOllama
from app.core import OLLAMA_MODEL_A, OLLAMA_MODEL_B, get_logger

logger = get_logger(__name__)

Profile = Literal["cold", "warm"]


def _profile_params(profile: Profile) -> dict:
    """
    Единые профили поведения модели под агента.
    Важно: температуру задаём ТОЛЬКО при создании ChatOllama.
    """
    if profile == "cold":
        return {"temperature": 0.15}
    elif profile == "warm":
        return {"temperature": 0.6}


@lru_cache(maxsize=16)
def get_chat_model(profile: Profile = "cold", num_ctx: int | None = None) -> ChatOllama:
    """
    Единая точка создания LLM-клиента, но с профилями.
    Кэш теперь на (profile, num_ctx), а не один инстанс на всё.
    """
    if profile == "cold":
        model = OLLAMA_MODEL_A
    else:
        model = OLLAMA_MODEL_B
    # params = _profile_params(profile)

    # Если хочешь явно контролировать контекст (часто полезно для агента)
    # if num_ctx is not None:
    #     params["num_ctx"] = num_ctx

    logger.info(
        f"Using LLM model: provider=ollama model={model} profile={profile}" # params={params}"
    )

    return ChatOllama(
        model=model,
        # **params,
    )