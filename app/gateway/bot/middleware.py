from aiogram import BaseMiddleware
from aiogram.dispatcher.event.bases import CancelHandler
from aiogram.types import Message, TelegramObject

from app.core import GROUP_ID
from app.core.logger import get_logger

logger = get_logger(__name__)

class OnlyGroupMiddleware(BaseMiddleware):
    async def __call__(self, handler, event: TelegramObject, data: dict):
        # пропускаем все, что не Message
        if not isinstance(event, Message):
            return await handler(event, data)

        # проверяем только команды
        if not event.text:
            return await handler(event, data)

        if event.chat.id != GROUP_ID:
            # можно ответить, но обычно в личке это будет уместно:
            logger.warning(f"ID чата: {event.chat.username} ({event.chat.id}). Сообщение: {event.text}")
            await event.answer("Эта команда работает только в опредленной группе.")
            return 

        return await handler(event, data)