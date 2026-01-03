from app.agent.state import State
from app.llm.prompt import build_memory_request_messages
from app.services.service import _extract_json
from app.services.service_db import build_memory_catalog, get_core_for_context, get_episodic_for_context, get_extended_for_context, normalize_memory_request

from app.core import get_logger

logger = get_logger(__name__)


def memory_read(llm):
    async def node(state: State) -> dict:
        # 1) Берём текущее сообщение пользователя (последнее в messages)
        user_message = ""
        try:
            user_message = str(state.get("messages", [])[-1].content)
        except Exception:
            user_message = ""

        # 2) Строим каталог памяти (меню возможностей)
        catalog = await build_memory_catalog()

        # 3) Просим модель вернуть JSON "что доставать"
        planner_messages = build_memory_request_messages(
            user_message=user_message,
            catalog=catalog,
        )

        raw_json = None
        try:
            response = await llm.ainvoke(planner_messages)
            raw_json = _extract_json(getattr(response, "content", "") or "")
            # logger.debug(f"Planner response JSON: {raw_json}")
        except Exception as e:
            logger.warning(f"Planner failed: {e}")
            raw_json = None

        # 4) Нормализуем запрос (режем лимиты, выкидываем неизвестные поля, fallback)
        req = normalize_memory_request(raw_json, catalog=catalog)

        # logger.debug(f"Memory request: {req}")

        # 5) Достаём память (SQL-фильтры -> candidates -> rerank embeddings)
        core_facts = await get_core_for_context(
            core_limit=50,
        )

        extended_facts = await get_extended_for_context(
            query_text=user_message,
            req=req,
            candidate_limit=200,
        )

        episodic_facts = await get_episodic_for_context(
            query_text=user_message,
            req=req,
            candidate_limit=300,
        )

        logger.info(f"Memory counts: core={len(core_facts)} ext={len(extended_facts)} epi={len(episodic_facts)}")

        # 6) Возвращаем в state. messages не трогаем.
        return {
            "core_facts": core_facts,
            "extended_facts": extended_facts,
            "episodic_facts": episodic_facts,
            # если хочешь дебажить — можно раскомментить:
            # "memory_request": req,
        }

    return node