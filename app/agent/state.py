from typing_extensions import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class State(TypedDict):
    # История сообщений. add_messages = "добавляй новые сообщения, не затирай список".
    messages: Annotated[list[BaseMessage], add_messages]