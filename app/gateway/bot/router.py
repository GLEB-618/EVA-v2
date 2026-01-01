from aiogram import Router
from aiogram.types import Message

from langchain_core.messages import HumanMessage

from app.core import get_logger


router = Router(name = "main")
logger = get_logger(__name__)


@router.message()
async def handle_message(message: Message, graph_app):
    logger.info(f"Received message from {message.chat.id}: {message.text}")

    text = message.text or ""

    out = await graph_app.ainvoke(
        {"messages": [HumanMessage(content=text)]},
        config={"configurable": {"thread_id": f"tg:{message.chat.id}"}},
    )

    await message.answer(out["messages"][-1].content)