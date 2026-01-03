import json
import re
from typing import Any, Optional
from langchain_core.messages import HumanMessage

from app.agent.state import State
from app.llm.prompt import build_memory_extraction_messages
from app.services.service_db import memory_read, memory_write as memory_write_service

from app.core import get_logger

logger = get_logger(__name__)


def _extract_json(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None

    # убираем ```json ... ```
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text.strip(),
        flags=re.IGNORECASE | re.MULTILINE,
    )

    # пытаемся найти первый объект {...}
    m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not m:
        return None

    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def memory_write(llm, history_window: int = 1000, max_facts: int = 6):
    async def node(state: State) -> dict:
        msgs = state["messages"]

        # 1) хвост истории (чтобы не жечь токены)
        tail = msgs[-history_window:] if len(msgs) > history_window else msgs

        # 2) для дедупа/контекста подмешаем релевантную память (как у тебя уже сделано)
        last_human = next((m for m in reversed(msgs) if isinstance(m, HumanMessage)), None)
        last_human_text = str(last_human.content) if last_human else ""
        data = await memory_read(last_human_text)

        # 3) промпт: "что сохранить" (модель вернёт JSON)
        mem_messages = build_memory_extraction_messages(
            tail,
            core_facts=data["core"],
            extended_facts=data["extended"],
            episodic_facts=data["episodic"],
        )

        resp = await llm.ainvoke(mem_messages)

        logger.info(f"Memory write response: {resp}")

        payload = _extract_json(getattr(resp, "content", "") or "")
        facts = (payload or {}).get("facts", [])
        if not isinstance(facts, list):
            facts = []

        facts = facts[:max_facts]
        seen = set()

        for f in facts:
            scope = str(f.get("scope", "episodic")).strip().lower()
            value = str(f.get("value", "")).strip()
            importance = float(f.get("importance", 0.5))

            if not value:
                continue
            if scope not in {"core", "extended", "episodic"}:
                scope = "episodic"
            importance = max(0.0, min(1.0, importance))

            key = (scope, value.lower())
            if key in seen:
                continue
            seen.add(key)

            await memory_write_service(scope, value, importance)

        logger.debug(f"Memory saved facts: {facts}")

        return {}

    return node