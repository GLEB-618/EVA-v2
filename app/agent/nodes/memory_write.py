from __future__ import annotations

import json
import re
from typing import Any, Optional

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

from app.agent.state import State
from app.core import get_logger
from app.db.session import session_factory
from app.llm.prompt import build_memory_write_messages
from app.services.service import _extract_json
from app.services.service_db import add_memory_fact, add_episodic_memory

logger = get_logger(__name__)


_ws = re.compile(r"\s+")


def _canon(s: str) -> str:
    """Нормализация для canonical_key."""
    s = (s or "").strip().lower()
    s = _ws.sub(" ", s)
    return s


def _make_canonical_key(tier: str, subject: str, predicate: str) -> str:
    return f"{_canon(tier)}|{_canon(subject)}|{_canon(predicate)}"


def _clamp01(x: Optional[float], default: Optional[float]) -> Optional[float]:
    if x is None:
        return default
    try:
        v = float(x)
    except Exception:
        return default
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _get_last_user_and_ai(messages: list[BaseMessage]) -> tuple[str, str]:
    """
    Берём последнюю HumanMessage и последнюю не-human (обычно AIMessage).
    Это безопаснее, чем [-1] и [-2], если есть tool messages.
    """
    user_text = ""
    ai_text = ""

    for m in reversed(messages):
        # human
        if m.__class__.__name__ == "HumanMessage" and not user_text:
            user_text = str(getattr(m, "content", "") or "")
        # assistant/ai (не human)
        if m.__class__.__name__ != "HumanMessage" and not ai_text:
            ai_text = str(getattr(m, "content", "") or "")

        if user_text and ai_text:
            break

    return user_text, ai_text


def memory_write(llm):
    async def node(state: State) -> dict:
        messages = state.get("messages", [])
        user_text, ai_text = _get_last_user_and_ai(messages)

        if not user_text and not ai_text:
            return {}

        # 1) Спросить у модели, что сохранять
        planner_msgs = build_memory_write_messages(
            user_text=user_text,
            ai_text=ai_text,
        )

        try:
            resp = await llm.ainvoke(planner_msgs)
            logger.debug(f"[memory_write] llm response: {getattr(resp, 'content', None)}")
            raw = _extract_json(getattr(resp, "content", "") or "")
            logger.debug(f"[memory_write] extracted json: {raw}")
        except Exception as e:
            logger.warning(f"[memory_write] llm invoke failed: {e}")
            return {}

        if not raw:
            logger.debug("[memory_write] no json extracted")
            return {}

        facts = raw.get("facts", [])
        episodic = raw.get("episodic", [])

        if not isinstance(facts, list):
            facts = []
        if not isinstance(episodic, list):
            episodic = []

        # 2) Санитизация (чтобы не записать мусор/сломанный формат)
        clean_facts: list[dict[str, Any]] = []
        for f in facts[:5]:
            if not isinstance(f, dict):
                continue

            tier = f.get("tier")
            subject = f.get("subject")
            predicate = f.get("predicate")
            value = f.get("value")

            if tier not in ("core", "extended"):
                continue
            if not isinstance(subject, str) or not subject.strip():
                continue
            if not isinstance(predicate, str) or not predicate.strip():
                continue
            if not isinstance(value, str) or not value.strip():
                continue

            confidence = _clamp01(f.get("confidence"), None)
            canonical_key = _make_canonical_key(tier, subject, predicate)

            clean_facts.append(
                {
                    "tier": tier,
                    "subject": subject.strip(),
                    "predicate": predicate.strip(),
                    "value": value.strip(),
                    "confidence": confidence,
                    "canonical_key": canonical_key,
                }
            )

        clean_episodes: list[dict[str, Any]] = []
        for ep in episodic[:2]:
            if not isinstance(ep, dict):
                continue

            event_type = ep.get("event_type")
            summary = ep.get("summary")
            content = ep.get("content")
            importance = _clamp01(ep.get("importance"), 0.0) or 0.0

            if not isinstance(event_type, str) or not event_type.strip():
                continue
            if not isinstance(summary, str) or not summary.strip():
                continue
            if not isinstance(content, str) or not content.strip():
                continue

            clean_episodes.append(
                {
                    "event_type": event_type.strip(),
                    "summary": summary.strip(),
                    "content": content.strip(),
                    "importance": float(importance),
                }
            )

        logger.debug(f"[memory_write] sanitized facts: {clean_facts}")
        logger.debug(f"[memory_write] sanitized episodic: {clean_episodes}")

        if not clean_facts and not clean_episodes:
            logger.debug("[memory_write] nothing to write after sanitize")
            return {}

        # 3) Запись в БД (upsert по canonical_key / по source ids если будут)
        # source ids (если ты когда-то добавишь в state)
        source_chat_id = state.get("source_chat_id")  # может отсутствовать
        source_message_id = state.get("source_message_id")

        await add_memory_fact(clean_facts)
        await add_episodic_memory(
            clean_episodes,
            source_chat_id=source_chat_id,
            source_message_id=source_message_id,
        )

        logger.debug(
            "[memory_write] wrote facts=%d episodic=%d",
            len(clean_facts),
            len(clean_episodes),
        )

        # Ничего не меняем в state, чтобы не ломать "последнее сообщение"
        return {}

    return node
