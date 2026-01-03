from typing import Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import Row, Sequence, delete, exists, select, or_, and_

from app.models.model import *

# Добавление Memory
async def insert_fact(session: AsyncSession, scope: str, value: str, importance: float|None = None) -> None:
    new = Memory(scope=scope, value=value, importance=importance)
    session.add(new)
    await session.flush()

# Чтение Memory
async def select_memory(session: AsyncSession) -> dict[str, list[Any]]:
    stmt = (
        select(
            Memory.value,
            Memory.scope,
            Memory.importance,
        )
        .order_by(Memory.importance.desc())
    )

    result = await session.execute(stmt)
    rows = result.all()  # [(value, scope, importance), ...]

    core_facts: list[str] = []
    extended_facts: list[dict[str, Any]] = []
    episodic_facts: list[dict[str, Any]] = []

    for value, scope, importance in rows:
        if scope == "core":
            core_facts.append(value)
        elif scope == "extended":
            extended_facts.append({"value": value, "importance": float(importance)})
        elif scope == "episodic":
            episodic_facts.append(value)

    return {"core": core_facts, "extended": extended_facts, "episodic": episodic_facts}

# async def insert_messages(session: AsyncSession, thread_id: int, role: str, content: str, name: str|None) -> None:
#     new = Messages(thread_id=thread_id, role=role, content=content, name=name)
#     session.add(new)
#     await session.flush()

# async def select_messages_by_thread(session: AsyncSession, thread_id: int, limit: int = 20) -> list[dict[str, Any]]:
#     stmt = select(Messages.role, Messages.content, Messages.name).where(Messages.thread_id == thread_id).order_by(Messages.id.desc())
#     result = await session.execute(stmt)
#     rows = result.all()  # [(role, content, name), ...]

#     result = await session.execute(stmt)
#     rows = list(result.all())
#     rows = list(reversed(rows))

#     result_lists = []
#     for role, content, name in rows:
#         if name == None:
#             result_lists.append({"role": role, "content": content})
#         else:
#             result_lists.append({"role": role, "name": name, "content": content})

#     return result_lists