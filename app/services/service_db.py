from typing import Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.db.session import session_factory
from app.llm.embedding import get_embedding_model, describe_embedding
from app.repository.repo import insert_fact, select_memory

from app.core import get_logger

logger = get_logger(__name__)


emb = get_embedding_model()
describe_embedding()
alpha = 0.7
beta = 0.3


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Приводим массив к [0, 1], аккуратно обрабатываем константный случай."""
    if arr.size == 0:
        return arr
    arr_min = float(arr.min())
    arr_max = float(arr.max())
    if arr_max - arr_min < 1e-8:
        # все одинаковые — пусть будет середина
        return np.full_like(arr, 0.5, dtype=float)
    return (arr - arr_min) / (arr_max - arr_min)

async def _hybrid_method(text: str, lists: list[dict[str, Any]]) -> list[str]:
    texts = [item["value"] for item in lists]
    importances = np.array([item["importance"] for item in lists], dtype=float)

    query_emb = await emb.encode([text])
    corpus_emb = await emb.encode(texts)

    sims = cosine_similarity(query_emb, corpus_emb)[0]

    sims_norm = _normalize(sims)
    imps_norm = _normalize(importances)

    scores = alpha * sims_norm + beta * imps_norm

    top_indices = scores.argsort()[::-1]
    top_texts = [texts[i] for i in top_indices]
    logger.debug("Топ факты (value, score): "+ repr([(texts[i], float(scores[i])) for i in top_indices]))

    return top_texts


async def memory_write(scope: str, value: str, importance: float):
    async with session_factory() as session:
        await insert_fact(session, scope, value, importance)
        await session.commit()
    
async def memory_read(text: str) -> dict[str, list]:
    async with session_factory() as session:
        data = await select_memory(session)
    extended_facts = await _hybrid_method(text, data["extended"]) if data["extended"] else []
    episodic_facts = await _hybrid_method(text, [{"value": v, "importance": 0.5} for v in data["episodic"]]) if data["episodic"] else []
    return {
        "core": data["core"],
        "extended": extended_facts,
        "episodic": episodic_facts,
    }