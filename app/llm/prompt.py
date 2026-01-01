from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import BaseMessage, SystemMessage


SYSTEM_BASE = """\
Ты — ассистент. Отвечай по делу, без лишней воды.
Если тебе не хватает данных — прямо скажи, что именно нужно уточнить.
Не выдумывай факты и результаты инструментов.
"""

def build_messages(
    messages: Sequence[BaseMessage],
    *,
    system: str = SYSTEM_BASE,
    memory: Optional[Dict[str, Any]] = None,
    keep_last: int = 24,
) -> List[BaseMessage]:
    """
    Собирает итоговый список сообщений для LLM:

    - 1 SystemMessage сверху (инструкции)
    - затем последние keep_last сообщений истории
    - опционально: добавляет "memory" в конец системного промпта текстовым блоком

    Как потом расширять:
    - сделай system промпты по режимам: system_blind/system_normal
    - в state держи mode и выбирай нужный system текст тут
    - memory можно форматировать красивее (facts/episodes/docs)
    - можно обрезать по токенам, а не по количеству сообщений
    """

    # 1) Трим истории (простой и надёжный)
    msgs = list(messages)
    if keep_last > 0 and len(msgs) > keep_last:
        msgs = msgs[-keep_last:]


    # 3) Собираем итоговый список
    return [SystemMessage(content=system), *msgs]