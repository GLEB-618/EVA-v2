from langchain_core.messages import HumanMessage

from app.agent.state import State
from app.llm.prompt import build_messages
from app.services.service_db import memory_read

from app.core import get_logger

logger = get_logger(__name__)


def chat(llm_tools):
    """
    llm_tools = модель (ChatOllama/OpenAI/etc) с bind_tools(...) или просто llm.
    Главное: чтобы у неё был async-метод .ainvoke(messages).
    """
    async def chat_node(state: State) -> dict:
        msgs = state["messages"]
        last_human = next((m for m in reversed(msgs) if isinstance(m, HumanMessage)), None)
        last_human_text = str(last_human.content) if last_human else "None"
        data = await memory_read(last_human_text)
        
        messages = build_messages(
            state["messages"],
            core_facts=data["core"],
            extended_facts=data["extended"],
            episodic_facts=data["episodic"]
        )

        ai_msg = await llm_tools.ainvoke(messages)

        logger.info(ai_msg)
        logger.info(f"Content: {getattr(ai_msg, 'content', None)}")
        logger.info(f"Tool calls: {getattr(ai_msg, 'tool_calls', None)}")

        return {"messages": [ai_msg]}

    return chat_node