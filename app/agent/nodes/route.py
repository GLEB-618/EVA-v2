from langchain_core.messages import HumanMessage

from app.agent.state import State
from app.llm.prompt import build_route_messages

from app.core import get_logger

logger = get_logger(__name__)


def route(llm_tools):
    async def route_node(state: State) -> dict:
        user_message = ""
        try:
            user_message = str(state.get("messages", [])[-1].content)
        except Exception:
            user_message = ""

        messages = build_route_messages(user_message)

        ai_msg = await llm_tools.ainvoke(messages)

        # logger.debug(ai_msg)
        logger.info(f"Content: {getattr(ai_msg, 'content', None)}")
        logger.info(f"Tool calls: {getattr(ai_msg, 'tool_calls', None)}")

        return {"route_to": str(getattr(ai_msg, 'content', "chat")).strip().lower()}

    return route_node