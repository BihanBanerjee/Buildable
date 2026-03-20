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
        user_prompt = state.get("user_prompt", "")
        api_key = configurable.get("openrouter_api_key")
        builder_model = state.get("builder_model", "google/gemini-2.5-pro")

        files_tracker: list = []

        # On follow-ups, emit builder_started (scaffold used to do this)
        if not is_first_message:
            safe_send_event(event_queue, {"e": "builder_started", "message": "Generating code for your application..."})

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

        # For follow-ups: pass the raw user prompt (no plan available)
        if not is_first_message:
            user_message = get_builder_prompt({"_user_prompt": user_prompt}, False)
        else:
            user_message = get_builder_prompt(plan, scaffold_ok)
        messages = [HumanMessage(content=user_message)]

        builder_error = None
        try:
            await asyncio.wait_for(
                stream_agent_events(agent_executor, messages, config, event_queue, project_id),
                timeout=180,
            )
        except asyncio.TimeoutError:
            builder_error = "Build timed out after 3 minutes. Please try a simpler prompt."
            print("Builder agent timed out after 3 minutes")
        except Exception as e:
            error_str = str(e)
            print(f"Builder agent error: {e}")
            traceback.print_exc()

            # Detect API credit/quota errors and surface them clearly
            if "402" in error_str or "credits" in error_str.lower() or "afford" in error_str.lower():
                builder_error = "Insufficient API credits. Please add credits to your API provider and try again."
            elif "429" in error_str or "rate" in error_str.lower():
                builder_error = "API rate limit reached. Please wait a moment and try again."
            elif "401" in error_str or "unauthorized" in error_str.lower():
                builder_error = "API authentication failed. Please check your API key."

        # Detect empty builds on first message (builder ran but created nothing)
        if is_first_message and not files_tracker and not builder_error:
            builder_error = "Build failed — no files were generated. This may be due to an API issue. Please try again."

        # If there was a critical error, send it to the user and abort
        if builder_error:
            safe_send_event(event_queue, {"e": "error", "message": builder_error})
            safe_send_event(event_queue, {"e": "builder_complete", "message": "Build phase complete"})

            await store_message(chat_id=project_id, role="assistant", content=builder_error, event_type="error")

            log_entry = timer.stop(status="error", error=builder_error)
            return {
                "files_created": [],
                "current_node": "builder",
                "error_message": builder_error,
                "success": False,
                "execution_log": [log_entry],
            }

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
            safe_send_event(event_queue, {"e": "error", "message": error_msg})

        log_entry = timer.stop(status="error", error=error_msg)
        return {
            "files_created": [],
            "current_node": "builder",
            "error_message": error_msg,
            "execution_log": [log_entry],
        }
