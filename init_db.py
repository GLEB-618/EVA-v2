from app.db.session import Base, engine
from app.models.model import *

async def init_db():
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)

    except Exception as e:
        print("Ошибка в пересоздании таблиц: %s", e)

if __name__ == "__main__":
    import asyncio
    asyncio.run(init_db())