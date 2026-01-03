from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from app.gateway.bot.middleware import OnlyGroupMiddleware
from app.gateway.bot.router import router
from app.core import BOT_TOKEN, get_logger

logger = get_logger(__name__)


async def start_telegram_bot(graph_app):
    try:
        bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
        dp = Dispatcher()

        dp.message.middleware(OnlyGroupMiddleware())
        dp.include_routers(router)

        logger.info("Starting Telegram bot")

        await dp.start_polling(bot, graph_app=graph_app)

    except Exception as e:
        logger.error(e)