import asyncio
from functools import lru_cache, partial
from concurrent.futures import ThreadPoolExecutor
from app.core import EMBEDDING_MODEL
from sentence_transformers import SentenceTransformer

from app.core import get_logger

logger = get_logger(__name__, "logs.log")


class EmbeddingModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._executor = ThreadPoolExecutor()
        self._model = SentenceTransformer(model_name)

    async def encode(self, inputs: list[str]):
        loop = asyncio.get_running_loop()
        func = partial(self._model.encode, inputs)

        return await loop.run_in_executor(self._executor, func)
    
    async def encode_one(self, text: str) -> list[float]:
        emb = await self.encode([text])  # shape = (1, dim)
        return emb[0].tolist()
    
@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    """
    Единая точка создания embedding-модели.
    """
    model = EMBEDDING_MODEL
    metricks = {
        "model": EMBEDDING_MODEL,
    }
    logger.info(f"Using embedding model: {metricks}")

    return EmbeddingModel(
        model_name=model,
    )