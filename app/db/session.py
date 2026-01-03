from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core import DATABASE_URL


engine = create_async_engine(
    url=DATABASE_URL,
    echo=False,
)

session_factory = async_sessionmaker(engine)


class Base(DeclarativeBase):
    pass
