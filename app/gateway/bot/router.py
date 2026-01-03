from aiogram import Router
from aiogram.types import Message

from langchain_core.messages import HumanMessage

from app.core import get_logger


router = Router(name = "main")
logger = get_logger(__name__)


@router.message()
async def handle_message(msg: Message, graph_app):
    if msg.text is None:
        return
    text = msg.text.strip()
    if msg.message_thread_id is None:
        await msg.reply("Пожалуйста, используйте темы для общения с ботом в группах.")
        return
    thread_id = msg.message_thread_id

    logger.info(f"Received message in thread {thread_id}: {text}")

    out = await graph_app.ainvoke(
        {"messages": [HumanMessage(content=text)]},
        config={"configurable": {"thread_id": f"tg:{thread_id}"}},
    )

    await msg.answer(out["messages"][-1].content)