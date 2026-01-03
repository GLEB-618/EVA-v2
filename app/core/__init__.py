from .config import settings as _settings
from .logger import get_logger


DATABASE_URL = _settings.DATABASE_URL
CHECKPOINT_DB_URI = _settings.CHECKPOINT_DB_URI

BOT_TOKEN = _settings.BOT_TOKEN
GROUP_ID = _settings.GROUP_ID

OLLAMA_MODEL = _settings.OLLAMA_MODEL
EMBEDDING_MODEL = _settings.EMBEDDING_MODEL

del _settings


__all__ = [
    "DATABASE_URL",
    "CHECKPOINT_DB_URI",
    "BOT_TOKEN",
    "GROUP_ID",
    "OLLAMA_MODEL",
    "EMBEDDING_MODEL",
    "get_logger",
]