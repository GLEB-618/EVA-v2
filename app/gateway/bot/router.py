from aiogram import Router
from aiogram.types import Message

from langchain_core.messages import HumanMessage

from app.core import get_logger
from app.services.service_download import download_photo


router = Router(name = "main")
logger = get_logger(__name__)


@router.message()
async def handle_message(msg: Message, graph_app):
    if msg.text is None and msg.caption is None:
        return
    if msg.bot is None:
        return
    
    text = msg.text.strip() if msg.text else msg.caption

    if msg.message_thread_id is None:
        await msg.reply("Пожалуйста, используйте темы для общения с ботом в группах.")
        return
    
    thread_id = msg.message_thread_id

    logger.info(f"Received message in thread {thread_id}: text={text}|photo={len(msg.photo) if msg.photo else 0}")

    photo = msg.photo[-1] if msg.photo else None
    if photo is not None:
        logger.info(f"Downloading photo {photo.file_id}...")
        data_url = await download_photo(msg.bot, photo)
        messages = HumanMessage(content=[
            {"type": "image_url", "image_url": data_url},
            {"type": "text", "text": text},
        ])
    else:
        messages = HumanMessage(content=text)

    out = await graph_app.ainvoke(
        {"messages": [messages]},
        config={"configurable": {"thread_id": f"tg:{thread_id}"}},
    )

    await msg.answer(out["messages"][-1].content)