from typing import Any, Optional
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import Row, Sequence, delete, desc, exists, select, or_, and_, func
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.models.model import *


async def add_memory_fact_repo(
    session: AsyncSession,
    *,
    tier: str,            # "core" | "extended"
    subject: str,
    predicate: str,
    value: str,
    canonical_key: str,
    confidence: Optional[float] = None,
) -> None:
    """
    Добавляет факт в MemoryFacts с дедупом по canonical_key.
    Если canonical_key уже существует — обновляет value/confidence/last_seen_at.
    """
    stmt = (
        pg_insert(MemoryFacts)
        .values(
            tier=tier,
            subject=subject,
            predicate=predicate,
            value=value,
            canonical_key=canonical_key,
            confidence=confidence,
        )
        .on_conflict_do_update(
            index_elements=["canonical_key"],
            set_={
                # если хочешь НЕ затирать value всегда — скажи, сделаем merge-политику
                "value": value,
                "confidence": confidence,
                "last_seen_at": datetime.now(timezone.utc),  # можно func.now(), но тут ок
            },
        )
    )

    await session.execute(stmt)
    await session.flush()

async def add_episodic_memory_repo(
    session: AsyncSession,
    *,
    event_type: str,
    summary: str,
    content: str,
    importance: float = 0.0,
    source_chat_id: Optional[int] = None,
    source_message_id: Optional[int] = None,
) -> None:
    """
    Добавляет эпизод в EpisodicMemory.
    Если переданы source_chat_id и source_message_id — делает дедуп по ним.
    Иначе просто вставляет новую запись.
    """
    values = dict(
        event_type=event_type,
        summary=summary,
        content=content,
        importance=importance,
        source_chat_id=source_chat_id,
        source_message_id=source_message_id,
    )

    # Если есть источник — делаем UPSERT (дедуп)
    if source_chat_id is not None and source_message_id is not None:
        stmt = (
            pg_insert(EpisodicMemory)
            .values(**values)
            .on_conflict_do_update(
                index_elements=["source_chat_id", "source_message_id"],
                set_={
                    # обычно эпизод один и тот же — можно обновить summary/content/importance
                    "event_type": event_type,
                    "summary": summary,
                    "content": content,
                    "importance": importance,
                    "last_seen_at": datetime.now(timezone.utc),
                },
            )
        )
        await session.execute(stmt)
        await session.flush()
        return

    # Если источника нет — просто вставляем
    new = EpisodicMemory(**values)
    session.add(new)
    await session.flush()

async def build_memory_catalog_repo(
    session: AsyncSession,
    *,
    predicates_limit: int = 20,
    subjects_limit: int = 50,
    event_types_limit: int = 20,
) -> dict[str, Any]:
    """
    Возвращает "каталог возможностей" памяти, без самих фактов/эпизодов.

    {
        "facts_catalog": {
        "subjects": [...],
        "predicates_top": [...],
        "counts": {"core": 12, "extended": 55}
        },
        "episodic_catalog": {
        "event_types": [...],
        "date_range": ["2025-12-01T12:00:00+00:00", "2026-01-03T10:00:00+00:00"] | None,
        "count": 210
        }
    }
    """

    # --- facts: counts by tier ---
    tier_counts_stmt = (
        select(MemoryFacts.tier, func.count(MemoryFacts.id))
        .group_by(MemoryFacts.tier)
    )
    tier_counts_rows = (await session.execute(tier_counts_stmt)).all()
    tier_counts: dict[str, int] = {tier: int(cnt) for tier, cnt in tier_counts_rows}

    # --- facts: distinct subjects ---
    subjects_stmt = (
        select(MemoryFacts.subject)
        .distinct()
        .order_by(MemoryFacts.subject.asc())
        .limit(subjects_limit)
    )
    subjects = [r[0] for r in (await session.execute(subjects_stmt)).all()]

    # --- facts: top predicates by frequency ---
    predicates_stmt = (
        select(MemoryFacts.predicate, func.count(MemoryFacts.id).label("cnt"))
        .group_by(MemoryFacts.predicate)
        .order_by(desc("cnt"), MemoryFacts.predicate.asc())
        .limit(predicates_limit)
    )
    predicates = [r.predicate for r in (await session.execute(predicates_stmt)).all()]

    facts_catalog = {
        "subjects": subjects,
        "predicates_top": predicates,
        "counts": {
            "core": tier_counts.get("core", 0),
            "extended": tier_counts.get("extended", 0),
        },
    }

    # --- episodic: total count ---
    epi_count_stmt = select(func.count(EpisodicMemory.id))
    epi_count = int((await session.execute(epi_count_stmt)).scalar_one())

    # --- episodic: date range ---
    epi_range_stmt = select(
        func.min(EpisodicMemory.created_at),
        func.max(EpisodicMemory.created_at),
    )
    epi_min_dt, epi_max_dt = (await session.execute(epi_range_stmt)).one()

    date_range = None
    if epi_min_dt is not None and epi_max_dt is not None:
        # оставляю datetime как есть; если хочешь строго строки — .isoformat()
        date_range = [epi_min_dt.isoformat(), epi_max_dt.isoformat()]

    # --- episodic: top event types by frequency ---
    event_types_stmt = (
        select(EpisodicMemory.event_type, func.count(EpisodicMemory.id).label("cnt"))
        .group_by(EpisodicMemory.event_type)
        .order_by(desc("cnt"), EpisodicMemory.event_type.asc())
        .limit(event_types_limit)
    )
    event_types = [r.event_type for r in (await session.execute(event_types_stmt)).all()]

    episodic_catalog = {
        "event_types": event_types,
        "date_range": date_range,
        "count": epi_count,
    }

    return {
        "facts_catalog": facts_catalog,
        "episodic_catalog": episodic_catalog,
    }

# ---------- READ CANDIDATES (SQL filters) ----------

async def select_core_facts_repo(
    session: AsyncSession,
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    stmt = (
        select(
            MemoryFacts.subject,
            MemoryFacts.predicate,
            MemoryFacts.value,
            MemoryFacts.confidence,
            MemoryFacts.last_seen_at,
        )
        .where(MemoryFacts.tier == "core")
        .order_by(MemoryFacts.confidence.desc())
        .limit(limit)
    )

    rows = (await session.execute(stmt)).all()
    return [
        {
            "subject": r.subject,
            "predicate": r.predicate,
            "value": r.value,
            "confidence": float(r.confidence) if r.confidence is not None else None,
            "last_seen_at": r.last_seen_at,
        }
        for r in rows
    ]


async def select_extended_candidates_repo(
    session: AsyncSession,
    *,
    subjects: Optional[list[str]] = None,
    predicates: Optional[list[str]] = None,
    min_confidence: Optional[float] = None,
    prefer_recent: bool = True,
    candidate_limit: int = 200,
) -> list[dict[str, Any]]:
    stmt = (
        select(
            MemoryFacts.subject,
            MemoryFacts.predicate,
            MemoryFacts.value,
            MemoryFacts.confidence,
            MemoryFacts.last_seen_at,
        )
        .where(MemoryFacts.tier == "extended")
    )

    if subjects:
        stmt = stmt.where(MemoryFacts.subject.in_(subjects))
    if predicates:
        stmt = stmt.where(MemoryFacts.predicate.in_(predicates))
    if min_confidence is not None:
        stmt = stmt.where(MemoryFacts.confidence >= float(min_confidence))

    if prefer_recent:
        stmt = stmt.order_by(MemoryFacts.last_seen_at.desc())
    else:
        stmt = stmt.order_by(MemoryFacts.confidence.desc().nullslast(), MemoryFacts.last_seen_at.desc())

    stmt = stmt.limit(candidate_limit)

    rows = (await session.execute(stmt)).all()
    return [
        {
            "subject": r.subject,
            "predicate": r.predicate,
            "value": r.value,
            "confidence": float(r.confidence) if r.confidence is not None else None,
            "last_seen_at": r.last_seen_at,
        }
        for r in rows
    ]


async def select_episodic_candidates_repo(
    session: AsyncSession,
    *,
    event_types: Optional[list[str]] = None,
    since_dt: Optional[datetime] = None,
    min_importance: Optional[float] = None,
    prefer_recent: bool = True,
    candidate_limit: int = 300,
) -> list[dict[str, Any]]:
    stmt = select(
        EpisodicMemory.event_type,
        EpisodicMemory.summary,
        EpisodicMemory.content,
        EpisodicMemory.importance,
        EpisodicMemory.created_at,
        # если у тебя есть last_seen_at — можно добавить:
        # EpisodicMemory.last_seen_at,
    )

    if event_types:
        stmt = stmt.where(EpisodicMemory.event_type.in_(event_types))
    if since_dt is not None:
        stmt = stmt.where(EpisodicMemory.created_at >= since_dt)
    if min_importance is not None:
        stmt = stmt.where(EpisodicMemory.importance >= float(min_importance))

    if prefer_recent:
        stmt = stmt.order_by(EpisodicMemory.created_at.desc(), EpisodicMemory.importance.desc())
    else:
        stmt = stmt.order_by(EpisodicMemory.importance.desc(), EpisodicMemory.created_at.desc())

    stmt = stmt.limit(candidate_limit)

    rows = (await session.execute(stmt)).all()
    return [
        {
            "event_type": r.event_type,
            "summary": r.summary,
            "content": r.content,
            "importance": float(r.importance) if r.importance is not None else 0.0,
            "created_at": r.created_at,
        }
        for r in rows
    ]