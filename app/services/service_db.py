from typing import Any, Optional
from datetime import datetime, timezone, timedelta

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.core import get_logger
from app.db.session import session_factory
from app.llm.embedding import get_embedding_model
from app.repository.repo import (
    add_episodic_memory_repo,
    add_memory_fact_repo,
    build_memory_catalog_repo,
    select_core_facts_repo,
    select_extended_candidates_repo,
    select_episodic_candidates_repo,
)

logger = get_logger(__name__)

emb = get_embedding_model()

# тюнинг потом
ALPHA_SIM = 0.7
BETA_WEIGHT = 0.2
GAMMA_RECENCY = 0.1


def _normalize(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    arr_min = float(arr.min())
    arr_max = float(arr.max())
    if arr_max - arr_min < 1e-8:
        return np.full_like(arr, 0.5, dtype=float)
    return (arr - arr_min) / (arr_max - arr_min)


def _dt_to_ts(dt: Optional[datetime]) -> float:
    if dt is None:
        return 0.0
    try:
        return float(dt.timestamp())
    except Exception:
        return 0.0


def _safe_bool(v: Any, default: bool) -> bool:
    return v if isinstance(v, bool) else default


def _safe_int(v: Any, default: int) -> int:
    try:
        if isinstance(v, bool):
            return default
        return int(v)
    except Exception:
        return default


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        if isinstance(v, bool):
            return None
        return float(v)
    except Exception:
        return None


async def build_memory_catalog() -> dict[str, Any]:
    async with session_factory() as session:
        result: dict[str, Any] = await build_memory_catalog_repo(session)
        logger.debug(f"Memory catalog: {result}")
        return result


def normalize_memory_request(
    data: Optional[dict[str, Any]],
    *,
    catalog: dict[str, Any],
) -> dict[str, Any]:
    """
    Делает safe-объект запроса к памяти:
    - режет лимиты
    - убирает неизвестные subject/predicate/event_type по каталогу
    - гарантирует наличие всех полей
    """
    out = {
        "extended": {
            "need": False,
            "k": 0,
            "subjects": [],
            "predicates": [],
            "min_confidence": None,
            "prefer_recent": True,
        },
        "episodic": {
            "need": False,
            "k": 0,
            "event_types": [],
            "since_days": None,
            "min_importance": None,
            "prefer_recent": True,
        },
    }

    facts_subjects = set(catalog.get("facts_catalog", {}).get("subjects", []) or [])
    facts_predicates = set(catalog.get("facts_catalog", {}).get("predicates_top", []) or [])
    epi_types = set(catalog.get("episodic_catalog", {}).get("event_types", []) or [])

    if not isinstance(data, dict):
        return out

    # extended
    ext = data.get("extended")
    if isinstance(ext, dict):
        out["extended"]["need"] = _safe_bool(ext.get("need"), out["extended"]["need"])
        out["extended"]["k"] = max(0, min(_safe_int(ext.get("k"), out["extended"]["k"]), 30))

        subs = ext.get("subjects", [])
        if isinstance(subs, list):
            out["extended"]["subjects"] = [s for s in subs if isinstance(s, str) and s in facts_subjects]

        preds = ext.get("predicates", [])
        if isinstance(preds, list):
            out["extended"]["predicates"] = [p for p in preds if isinstance(p, str) and p in facts_predicates]

        out["extended"]["min_confidence"] = _safe_float(ext.get("min_confidence"))
        out["extended"]["prefer_recent"] = _safe_bool(ext.get("prefer_recent"), out["extended"]["prefer_recent"])

        if out["extended"]["k"] > 0 and (out["extended"]["subjects"] or out["extended"]["predicates"]):
            out["extended"]["need"] = True

        if not out["extended"]["need"]:
            out["extended"]["k"] = 0
            out["extended"]["subjects"] = []
            out["extended"]["predicates"] = []

    # episodic
    epi = data.get("episodic")
    if isinstance(epi, dict):
        out["episodic"]["need"] = _safe_bool(epi.get("need"), out["episodic"]["need"])
        out["episodic"]["k"] = max(0, min(_safe_int(epi.get("k"), out["episodic"]["k"]), 15))

        ets = epi.get("event_types", [])
        if isinstance(ets, list):
            out["episodic"]["event_types"] = [t for t in ets if isinstance(t, str) and t in epi_types]

        sd = epi.get("since_days")
        out["episodic"]["since_days"] = _safe_int(sd, 0) if isinstance(sd, (int, float)) else None

        out["episodic"]["min_importance"] = _safe_float(epi.get("min_importance"))
        out["episodic"]["prefer_recent"] = _safe_bool(epi.get("prefer_recent"), out["episodic"]["prefer_recent"])

        if out["episodic"]["k"] > 0 and out["episodic"]["event_types"]:
            out["episodic"]["need"] = True

        if not out["episodic"]["need"]:
            out["episodic"]["k"] = 0
            out["episodic"]["event_types"] = []

    return out


async def hybrid_rank_facts(
    query_text: str,
    facts: list[dict[str, Any]],
    *,
    k: int,
    alpha: float = ALPHA_SIM,
    beta: float = BETA_WEIGHT,
    gamma: float = GAMMA_RECENCY,
) -> list[dict[str, Any]]:
    if not facts or k <= 0:
        return []

    texts = [f"{it.get('predicate','')}: {it.get('value','')}" for it in facts]
    confidences = np.array(
        [float(it["confidence"]) if it.get("confidence") is not None else 0.5 for it in facts],
        dtype=float,
    )
    recencies = np.array([_dt_to_ts(it.get("last_seen_at")) for it in facts], dtype=float)

    query_emb = await emb.encode([query_text])
    corpus_emb = await emb.encode(texts)
    sims = cosine_similarity(query_emb, corpus_emb)[0]

    sims_norm = _normalize(np.asarray(sims, dtype=float))
    conf_norm = _normalize(confidences)
    rec_norm = _normalize(recencies)

    scores = alpha * sims_norm + beta * conf_norm + gamma * rec_norm
    logger.debug(f"{query_text}, {sims_norm}, {conf_norm}, {rec_norm}, {scores}")
    top_idx = scores.argsort()[::-1][:k]
    return [facts[i] for i in top_idx]


async def hybrid_rank_episodic(
    query_text: str,
    episodes: list[dict[str, Any]],
    *,
    k: int,
    alpha: float = ALPHA_SIM,
    beta: float = BETA_WEIGHT,
    gamma: float = GAMMA_RECENCY,
) -> list[dict[str, Any]]:
    if not episodes or k <= 0:
        return []

    texts = [str(it.get("summary", "")) for it in episodes]
    importances = np.array([float(it.get("importance", 0.0)) for it in episodes], dtype=float)
    recencies = np.array([_dt_to_ts(it.get("created_at")) for it in episodes], dtype=float)

    query_emb = await emb.encode([query_text])
    corpus_emb = await emb.encode(texts)
    sims = cosine_similarity(query_emb, corpus_emb)[0]

    sims_norm = _normalize(np.asarray(sims, dtype=float))
    imp_norm = _normalize(importances)
    rec_norm = _normalize(recencies)

    scores = alpha * sims_norm + beta * imp_norm + gamma * rec_norm
    logger.debug(f"{query_text}, {sims_norm}, {imp_norm}, {rec_norm}, {scores}")
    top_idx = scores.argsort()[::-1][:k]
    return [episodes[i] for i in top_idx]


# ---------- 3 ОБЁРТКИ ДЛЯ memory_read ----------

async def get_core_for_context(
    *,
    core_limit: int = 50,
) -> list[dict[str, Any]]:
    async with session_factory() as session:
        return await select_core_facts_repo(session, limit=core_limit)


async def get_extended_for_context(
    *,
    query_text: str,
    req: dict[str, Any],
    candidate_limit: int = 200,
) -> list[dict[str, Any]]:
    ext = req["extended"]
    if not ext["need"] or ext["k"] <= 0:
        return []

    async with session_factory() as session:
        candidates = await select_extended_candidates_repo(
            session,
            subjects=ext["subjects"] or None,
            predicates=ext["predicates"] or None,
            min_confidence=ext["min_confidence"],
            prefer_recent=ext["prefer_recent"],
            candidate_limit=candidate_limit,
        )

    # если запроса по смыслу нет — просто берем первые k (они уже отсортированы SQL-ом)
    if not query_text.strip():
        return candidates[: ext["k"]]

    return await hybrid_rank_facts(query_text, candidates, k=ext["k"])


async def get_episodic_for_context(
    *,
    query_text: str,
    req: dict[str, Any],
    candidate_limit: int = 300,
) -> list[dict[str, Any]]:
    epi = req["episodic"]
    if not epi["need"] or epi["k"] <= 0:
        return []

    since_dt = None
    if epi["since_days"] is not None and epi["since_days"] > 0:
        since_dt = datetime.now(timezone.utc) - timedelta(days=int(epi["since_days"]))

    async with session_factory() as session:
        candidates = await select_episodic_candidates_repo(
            session,
            event_types=epi["event_types"] or None,
            since_dt=since_dt,
            min_importance=epi["min_importance"],
            prefer_recent=epi["prefer_recent"],
            candidate_limit=candidate_limit,
        )

    if not query_text.strip():
        return candidates[: epi["k"]]

    return await hybrid_rank_episodic(query_text, candidates, k=epi["k"])


async def add_memory_fact(clean_facts: list[dict[str, Any]]) -> None:
    async with session_factory() as session:
        for f in clean_facts:
            await add_memory_fact_repo(
                session,
                tier=f["tier"],
                subject=f["subject"],
                predicate=f["predicate"],
                value=f["value"],
                canonical_key=f["canonical_key"],
                confidence=f["confidence"],
            )
        await session.commit()

async def add_episodic_memory(clean_episodes: list[dict[str, Any]], source_chat_id, source_message_id) -> None:
    async with session_factory() as session:
        for ep in clean_episodes:
            await add_episodic_memory_repo(
                session,
                event_type=ep["event_type"],
                summary=ep["summary"],
                content=ep["content"],
                importance=ep["importance"],
                source_chat_id=source_chat_id if isinstance(source_chat_id, int) else None,
                source_message_id=source_message_id if isinstance(source_message_id, int) else None,
            )
        await session.commit()
    