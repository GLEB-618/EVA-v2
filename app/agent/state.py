from typing import Any
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class State(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]

    # память (перезаписывается на каждом read)
    core_facts: list[dict[str, Any]]
    extended_facts: list[dict[str, Any]]
    episodic_facts: list[dict[str, Any]]

    # опционально для дебага
    memory_request: dict[str, Any]
    memory_catalog: dict[str, Any]