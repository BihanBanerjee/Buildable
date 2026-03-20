"""
Node 3: BUILDER — LLM creates component and page files via ReAct agent.

Graceful degradation: if scaffold failed, builder gets the full follow_up tool set
(including execute_command) so it can install deps and write App.jsx itself.
"""

import asyncio
import traceback

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent

from ..graph_state import GraphState
from ..agent import create_llm
from ..tools import create_tools
from ..prompts import (
    BUILDER_SYSTEM_FIRST,
    BUILDER_SYSTEM_FOLLOWUP,
    BUILDER_SYSTEM_FALLBACK,
    get_builder_prompt,
)
from ..formatters import generate_build_summary
from .helpers import safe_send_event, store_message, stream_agent_events, NodeTimer


async def builder_node(state: GraphState, config: RunnableConfig) -> dict:
    """Run the builder ReAct agent to generate component and page code.

    For first builds: only creates component/page files (scaffold handles App.jsx + deps).
    For follow-ups: full builder with read/write/exec capabilities.

    Graceful degradation: if scaffold_complete is False, the builder gets the full
    follow_up tool set so it can handle deps/routing itself.
    """
    timer = NodeTimer("builder")
    configurable = config.get("configurable", {})
    event_queue = configurable.get("event_queue")
    sandbox = configurable.get("sandbox")

    try:
        if not sandbox:
            raise Exception("Sandbox not available")

        is_first_message = state.get("is_first_message", True)
        scaffold_ok = state.get("scaffold_complete", False)
        plan = state.get("plan", {})
        project_id = state.get("project_id", "")
        api_key = configurable.get("openrouter_api_key")
        builder_model = state.get("builder_model", "google/gemini-2.5-pro")

        files_tracker: list = []

        # ── Graceful degradation ──
        # If scaffold succeeded on a first build → narrow tool set (create only).
        # If scaffold failed → give the builder the full tool set so it can self-recover.
        if is_first_message and scaffold_ok:
            tool_mode = "first_build"
            system_prompt = BUILDER_SYSTEM_FIRST
        elif is_first_message and not scaffold_ok:
            tool_mode = "first_build_fallback"
            system_prompt = BUILDER_SYSTEM_FALLBACK
            print("⚠ Scaffold failed — builder using fallback mode for full self-recovery")
        else:
            tool_mode = "follow_up"
            system_prompt = BUILDER_SYSTEM_FOLLOWUP

        tools = create_tools(
            sandbox, event_queue, project_id,
            files_tracker=files_tracker,
            mode=tool_mode,
        )

        builder_llm = create_llm(api_key, builder_model, max_tokens=16000)

        agent_executor = create_react_agent(
            builder_llm,
            tools,
            prompt=SystemMessage(content=system_prompt),
        )

        user_message = get_builder_prompt(plan, is_first_message and scaffold_ok)
        messages = [HumanMessage(content=user_message)]

        try:
            await asyncio.wait_for(
                stream_agent_events(agent_executor, messages, config, event_queue, project_id),
                timeout=180,
            )
        except asyncio.TimeoutError:
            print("Builder agent timed out after 3 minutes")
        except Exception as e:
            print(f"Builder agent error: {e}")
            traceback.print_exc()

        # Emit build summary
        summary = generate_build_summary(files_tracker, [], plan)
        safe_send_event(event_queue, {"e": "summary", "message": summary})
        safe_send_event(event_queue, {"e": "builder_complete", "message": "Build phase complete"})

        await store_message(chat_id=project_id, role="assistant", content=summary, event_type="summary")

        log_entry = timer.stop(files_created=files_tracker)
        return {
            "files_created": files_tracker,
            "current_node": "builder",
            "execution_log": [log_entry],
        }

    except Exception as e:
        error_msg = f"Builder error: {str(e)}"
        print(error_msg)
        traceback.print_exc()

        if event_queue:
            safe_send_event(event_queue, {"e": "builder_error", "message": error_msg})

        log_entry = timer.stop(status="error", error=error_msg)
        return {
            "files_created": [],
            "current_node": "builder",
            "error_message": error_msg,
            "execution_log": [log_entry],
        }
