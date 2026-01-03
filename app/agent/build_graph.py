from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from app.agent.nodes.memory_read import memory_read
from app.agent.state import State
from app.agent.nodes.chat import chat
from app.agent.nodes.memory_write import memory_write
from app.tools.time_tools import now

from app.core import get_logger

logger = get_logger(__name__)


def build_graph(llm: ChatOllama, checkpointer=None):

    tools = [now]
    llm_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    builder = StateGraph(State)

    builder.add_node("memory_read", memory_read(llm))
    builder.add_node("chat", chat(llm_tools))
    builder.add_node("tools", tool_node)
    builder.add_node("memory_write", memory_write(llm))

    builder.add_edge(START, "memory_read")
    builder.add_edge("memory_read", "chat")
    builder.add_conditional_edges("chat", tools_condition, {"tools": "tools", "__end__": "memory_write"},)
    builder.add_edge("tools", "chat")
    builder.add_edge("chat", "memory_write")
    builder.add_edge("memory_write", END)

    logger.info("Graph built successfully.")

    return builder.compile(checkpointer=checkpointer)