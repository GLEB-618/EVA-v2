from typing import Any, Dict, List, Optional, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

from app.core import get_logger

logger = get_logger(__name__)


SYSTEM_BASE = """Ты — ассистент. Отвечай по делу, без лишней воды.
Если тебе не хватает данных — прямо скажи, что именно нужно уточнить.
Не выдумывай факты и результаты инструментов."""


MEMORY_EXTRACT_SYSTEM = """Ты — модуль записи памяти ассистента.

По истории диалога выдели факты, которые стоит сохранить в память.

Сохраняй:
- устойчивые факты о пользователе (предпочтения, навыки, проекты, цели)
- важные решения/события (что-то, что пригодится позже)

Не сохраняй:
- болтовню, приветствия, одноразовые фразы
- догадки и предположения"""


def _fmt_facts(title: str, facts: list, limit: int = 20) -> str:
    if not facts:
        return f"[{title}]\n- (empty)\n"
    lines = [f"- {f['value']}" if isinstance(f, dict) else f"- {str(f)}"
             for f in facts[:limit]]
    return f"[{title}]\n" + "\n".join(lines) + "\n"

def build_messages(messages: Sequence[BaseMessage], core_facts = None, extended_facts = None, episodic_facts = None) -> List[BaseMessage]:
    core_facts = core_facts or []
    extended_facts = extended_facts or []
    episodic_facts = episodic_facts or []

    memory_block = (
        "ПАМЯТЬ (служебный контекст, только для использования внутри)\n"
        "Правила: используй как фон для ответа. Не упоминай и не цитируй, если пользователь прямо не спрашивает.\n"
        "Если факты противоречат друг другу — приоритет более свежим эпизодическим фактам.\n\n"
        + _fmt_facts("CORE", core_facts)
        + _fmt_facts("EXTENDED", extended_facts)
        + _fmt_facts("EPISODIC", episodic_facts)
    )

    # logger.debug("Built memory block:\n" + memory_block)

    msgs = list(messages)

    return [
        SystemMessage(content=SYSTEM_BASE),
        SystemMessage(content=memory_block),
        *msgs
    ]

def build_memory_extraction_messages(messages: Sequence[BaseMessage], core_facts = None, extended_facts = None, episodic_facts = None) -> List[BaseMessage]:
    core_facts = core_facts or []
    extended_facts = extended_facts or []
    episodic_facts = episodic_facts or []

    memory_block = (
        "Это уже известные тебе факты из памяти агента. НЕ ДОБАВЛЯЙ их снова.\n"
        + _fmt_facts("CORE", core_facts)
        + _fmt_facts("EXTENDED", extended_facts)
        + _fmt_facts("EPISODIC", episodic_facts)
    )

    msgs = list(messages)

    return [
        SystemMessage(content=MEMORY_EXTRACT_SYSTEM),
        *msgs,
        HumanMessage(content=(
            "Выдели из истории диалога факты для сохранения в память агента.\n" +
            memory_block +
            "\nВерни СТРОГО JSON без текста вокруг в формате:\n"
            '{"facts":[{"scope":"core|extended|episodic","value":"...","importance":0.0}]}\n'
            "Если нечего сохранять: {\"facts\":[]}"
        ))
    ]

    # return [
    #     SystemMessage(content=MEMORY_EXTRACT_SYSTEM),
    #     SystemMessage(content=memory_block),
    #     *msgs,
    #     HumanMessage(content=(
    #         "Верни СТРОГО JSON без текста вокруг в формате:\n"
    #         '{"facts":[{"scope":"core|extended|episodic","value":"...","importance":0.0}]}\n'
    #         "Если нечего сохранять: {\"facts\":[]}"
    #     ))
    # ]