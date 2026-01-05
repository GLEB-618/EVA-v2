import json
from typing import Any, Optional


def _extract_json(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None

    s = text.strip()

    # убираем ```json ... ```
    if s.startswith("```"):
        s = s.strip("`")
        s = s.replace("json", "", 1).strip()

    decoder = json.JSONDecoder()

    # ищем первую "{" и пытаемся raw_decode оттуда
    start = s.find("{")
    if start == -1:
        return None

    try:
        obj, _end = decoder.raw_decode(s[start:])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None