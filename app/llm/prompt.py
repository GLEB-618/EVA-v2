from typing import Any
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

from app.core import get_logger

logger = get_logger(__name__)


SYSTEM_BASE = """Ты — ассистент. Отвечай по делу, без лишней воды. Не используй Markdown.
Если тебе не хватает данных — прямо скажи, что именно нужно уточнить.
Не выдумывай факты и результаты инструментов."""

MEMORY_PLANNER_SYSTEM = """Ты — Memory Planner. Твоя задача: по сообщению пользователя решить, какие данные нужно ДОСТАТЬ из памяти перед ответом.

Входные данные:
1) MEMORY_CATALOG — каталог того, что вообще есть в памяти (subjects, predicates_top, event_types, counts, date_range).
2) USER_MESSAGE — текущее сообщение пользователя.

Выход:
- Верни СТРОГО один валидный JSON-объект и НИЧЕГО больше (без текста, без пояснений, без markdown).
- Используй только те значения subject/predicate/event_type, которые есть в MEMORY_CATALOG.
- Не запрашивай слишком много: extended.k <= 30, episodic.k <= 15.
- Core факты обычно маленькие — чаще ставь need_core=true, но если они точно не нужны, можешь need_core=false.
- Episodic проси только если важен контекст/история/ошибки/предыдущие решения.

JSON-схема (строго эти поля):
{
  "extended": {
    "need": boolean,
    "k": integer,
    "subjects": string[],
    "predicates": string[],
    "min_confidence": number|null,
    "prefer_recent": boolean
  },
  "episodic": {
    "need": boolean,
    "k": integer,
    "event_types": string[],
    "since_days": integer|null,
    "min_importance": number|null,
    "prefer_recent": boolean
  }
}

Правила заполнения:
- Если extended.need=false → всё равно верни extended объект, но оставь списки пустыми и k=0.
- Если episodic.need=false → аналогично: списки пустые и k=0.
- Если пользователь спрашивает “вспомни/ты помнишь/раньше/что мы решили” → episodic.need=true.
- Если вопрос про настройки/предпочтения/железо/проекты → extended.need=true и укажи нужные predicates/subjects.
- prefer_recent=true, если важна актуальность/последнее состояние."""

MEMORY_WRITE_SYSTEM = """Ты — Memory Writer (сохранение памяти).
Твоя задача: по последнему сообщению пользователя и последнему ответу ассистента выделить, что стоит сохранить в долговременной памяти.

Правила:
- Сохраняй ТОЛЬКО устойчивые факты (предпочтения, железо/софт, проекты, договорённости) и важные эпизоды (решение/вывод/ошибка/важное событие).
- Не сохраняй приветствия, болтовню, одноразовые детали.
- Facts: максимум 5.
- Episodic: максимум 2.
- Верни СТРОГО один валидный JSON-объект. Если ты решаешь что-то не сохранить, то объясни причину.

Схема JSON:
{
  "facts": [
    {
      "tier": "core" | "extended",
      "subject": "строка",
      "predicate": "строка",
      "value": "строка",
      "confidence": number|null
    }
  ],
  "episodic": [
    {
      "event_type": "строка",
      "summary": "коротко (1-2 предложения)",
      "content": "подробности/контекст",
      "importance": number
    }
  ]
}

Подсказки:
- tier="core" только для очень стабильных личных фактов пользователя (редко).
- tier="extended" для настроек/железа/предпочтений/проектов.
- importance в диапазоне 0..1 (0.7+ только если реально важно).
- subject="Пользователь", если факт про самого пользователя.
- Предпочтения стиля общения (тон, род, язык, формат ответов) — сохранять как core."""

ROUTER_SYS = """Ты — роутер для агента.
Ты видишь USER_MESSAGE и все текущие tools, которые доступны ассистенту. Тебе КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО использовать tools. Твоя задача — только решить, какой путь выбрать: "tools" или "chat", исходя из USER_MESSAGE.
Верни СТРОГО строку:
tools|chat
Правила:
- tools если ты понимаешь, что в будущем нужно использовать инструменты.
- chat если это болтовня, объяснение, мнение, совет, идеи без внешних действий.
"""


def _fmt_facts(title: str, facts: list) -> str:
    if not facts:
        return f"{title}: []\n"
    
    lines = []

    for f in facts:
        if title == "core_facts" or title == "extended_facts":
            if isinstance(f, dict):
                sub = f.get("subject", "(unknown subject)")
                pred = f.get("predicate", "(unknown predicate)")
                val = f.get("value", "(no value)")
                lines.append({"subject": sub, "predicate": pred, "value": val})
    return f"{title}: {lines}\n"

def build_messages(messages: list[BaseMessage], core_facts = None, extended_facts = None, episodic_facts = None) -> list[BaseMessage]:
    core_facts = core_facts or []
    extended_facts = extended_facts or []
    episodic_facts = episodic_facts or []

    memory_block = (
        "ПАМЯТЬ (служебный контекст, только для использования внутри)\n"
        "Правила: используй как фон для ответа. Не упоминай и не цитируй, если пользователь прямо не спрашивает.\n"
        "Если факты противоречат друг другу — приоритет более свежим эпизодическим фактам.\n\n"
        + _fmt_facts('core_facts', core_facts)
        + _fmt_facts("extended_facts", extended_facts)
        + _fmt_facts("episodic_facts", episodic_facts)
    )

    logger.debug("Built memory block:\n" + memory_block)

    return [
        SystemMessage(content=SYSTEM_BASE),
        SystemMessage(content=memory_block),
        *messages
    ]

def build_memory_request_messages(user_message: str, catalog: dict[str, Any]) -> list:
    # logger.debug(f"Building memory request messages with user_message: {user_message} and catalog: {catalog}")
    return [
        SystemMessage(content=MEMORY_PLANNER_SYSTEM),
        HumanMessage(content=f"MEMORY_CATALOG:\n{catalog}\n\nUSER_MESSAGE:\n{user_message}"),
    ]

def build_memory_write_messages(user_text: str, ai_text: str, core_facts = None, extended_facts = None, episodic_facts = None) -> list:
    core_facts = core_facts or []
    extended_facts = extended_facts or []
    episodic_facts = episodic_facts or []
    # logger.debug(f"Building memory write messages with user_text: {user_text} and ai_text: {ai_text}")
    memory_block = (
        "ПАМЯТЬ (служебный контекст, только для использования внутри)\n"
        "Правила: для понимания, что уже известно.\n\n"
        + _fmt_facts('core_facts', core_facts)
        + _fmt_facts("extended_facts", extended_facts)
        + _fmt_facts("episodic_facts", episodic_facts)
    )
    logger.debug("Built memory block for write:\n" + memory_block)
    return [
        SystemMessage(content=MEMORY_WRITE_SYSTEM),
        SystemMessage(content=memory_block),
        HumanMessage(content=f"USER_MESSAGE:\n{user_text}\n\nASSISTANT_ANSWER:\n{ai_text}"),
    ]

def build_route_messages(user_message: str) -> list:
    return [
        SystemMessage(content=ROUTER_SYS),
        HumanMessage(content=f"USER_MESSAGE:\n{user_message}"),
    ]