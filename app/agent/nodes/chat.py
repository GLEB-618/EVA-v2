from langchain_core.messages import HumanMessage

from app.agent.state import State
from app.llm.prompt import build_messages

from app.core import get_logger

logger = get_logger(__name__)


def chat(llm_tools):
    async def chat_node(state: State) -> dict:
        logger.debug(f"State: {state}")
        
        messages = build_messages(
            state.get("messages", []),
            core_facts=state.get("core_facts", []),
            extended_facts=state.get("extended_facts", []),
            episodic_facts=state.get("episodic_facts", [])
        )

        ai_msg = await llm_tools.ainvoke(messages)

        # logger.debug(ai_msg)
        logger.info(f"Content: {getattr(ai_msg, 'content', None)}")
        logger.info(f"Tool calls: {getattr(ai_msg, 'tool_calls', None)}")

        return {"messages": [ai_msg]}

    return chat_node