import base64
from io import BytesIO
from aiogram import Bot
from aiogram.types import PhotoSize


def bytes_to_data_url(raw: bytes, mime: str = "image/jpeg") -> str:
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"

async def download_photo(bot: Bot, photo: PhotoSize) -> str:
    photo_file = await bot.get_file(photo.file_id)

    buf = BytesIO()

    await bot.download(photo_file, destination=buf)

    raw = buf.getvalue()
    data_url = bytes_to_data_url(raw, mime="image/jpeg")

    return data_url