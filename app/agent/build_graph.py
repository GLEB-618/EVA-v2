from langgraph.graph import StateGraph, START, END
from app.agent.state import State
from app.agent.nodes.node import make_chat_node

from app.core import get_logger

logger = get_logger(__name__)



def build_graph(llm_tools, *, checkpointer=None):
    builder = StateGraph(State)

    builder.add_node("chat", make_chat_node(llm_tools))

    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)

    logger.info("Graph built successfully.")

    return builder.compile(checkpointer=checkpointer)