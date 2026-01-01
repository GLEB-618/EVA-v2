from app.agent.state import State
from app.llm.prompt import build_messages  


def make_chat_node(llm_tools):
    """
    llm_tools = модель (ChatOllama/OpenAI/etc) с bind_tools(...) или просто llm.
    Главное: чтобы у неё был async-метод .ainvoke(messages).
    """
    async def chat_node(state: State) -> dict:
        print("Chat node invoked.")
        print(state)
        # Собираем system + последние сообщения (+ опциональная память)
        messages = build_messages(
            state["messages"],
            memory=None,
            keep_last=24,
        )

        # Асинхронный вызов модели (НЕ блокирует event loop)
        ai_msg = await llm_tools.ainvoke(messages)

        # Возвращаем как patch state: LangGraph добавит это в историю messages
        return {"messages": [ai_msg]}
    
    print("Chat node created.")
    print(llm_tools)

    return chat_node