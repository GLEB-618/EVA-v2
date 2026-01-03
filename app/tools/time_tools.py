from datetime import datetime
from zoneinfo import ZoneInfo
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from app.core import get_logger

logger = get_logger(__name__)


class NowInput(BaseModel):
    tz: str = Field(
        default="Europe/Moscow",
        description="IANA timezone, например 'Europe/Moscow'. Если не задано — Europe/Moscow.",
    )
    fmt: str = Field(
        default="%Y-%m-%d %H:%M:%S %Z",
        description="Формат вывода времени в стиле strftime.",
    )


@tool(args_schema=NowInput)
def now(tz: str = "Europe/Moscow", fmt: str = "%Y-%m-%d %H:%M:%S %Z") -> str:
    """
    Возвращает текущее локальное время в указанном часовом поясе.
    Возвращает: str
    """
    dt = datetime.now(ZoneInfo(tz))
    logger.debug(f"Getting current time for timezone: {tz} with format: {fmt} | Current time: {dt.strftime(fmt)}")
    return dt.strftime(fmt)