import asyncio

from app.agent.build_graph import build_graph
from app.gateway.bot.bot import start_telegram_bot
from app.llm.client import get_chat_model, describe_llm

async def main():
    llm = get_chat_model(temperature=0.0)
    describe_llm()
    graph_app = build_graph(llm, checkpointer=None)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(start_telegram_bot(graph_app))

if __name__ == "__main__":
    asyncio.run(main())