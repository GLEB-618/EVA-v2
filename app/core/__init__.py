from .config import settings as _settings
from .logger import get_logger

BOT_TOKEN = _settings.BOT_TOKEN
OLLAMA_MODEL = _settings.OLLAMA_MODEL
del _settings

__all__ = [
    "BOT_TOKEN",
    "OLLAMA_MODEL",
    "get_logger",
]