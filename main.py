import sys
import asyncio

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.agent.build_graph import build_graph
from app.core import CHECKPOINT_DB_URI as DB_URI
from app.gateway.bot.bot import start_telegram_bot


async def main():
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.setup()

        graph_app = build_graph(checkpointer=checkpointer)

        png_bytes = graph_app.get_graph().draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(png_bytes)

        async with asyncio.TaskGroup() as tg:
            tg.create_task(start_telegram_bot(graph_app))

if __name__ == "__main__":
    asyncio.run(main())