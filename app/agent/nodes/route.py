from langchain_core.messages import HumanMessage

from app.agent.state import State
from app.llm.prompt import build_route_messages

from app.core import get_logger

logger = get_logger(__name__)


def route(llm_tools):
    async def route_node(state: State) -> dict:
        user_message = ""
        try:
            messages = state.get("messages", [])[-1]
            if type(messages.content) == str:
                user_message = str(messages.content)
            elif type(messages.content) == list:
                return {"route_to": "tools"}
        except Exception:
            user_message = ""

        messages = build_route_messages(user_message)

        ai_msg = await llm_tools.ainvoke(messages)

        # logger.debug(ai_msg)
        logger.info(f"Content: {getattr(ai_msg, 'content', None)}")
        logger.info(f"Tool calls: {getattr(ai_msg, 'tool_calls', None)}")

        return {"route_to": str(getattr(ai_msg, 'content', "chat")).strip().lower()}

    return route_node