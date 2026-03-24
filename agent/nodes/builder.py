"""
Node 3: BUILDER — LLM creates component and page files.

First builds use single-shot generation (1 LLM call → batch file write).
Follow-ups use the ReAct agent for surgical edits.
Graceful degradation: if scaffold failed, builder gets the full tool set.
"""

import asyncio
import traceback

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent

from ..graph_state import GraphState
from ..agent import create_llm, get_fast_model
from ..tools import create_tools
from ..prompts import (
    BUILDER_SYSTEM_FIRST,
    BUILDER_SYSTEM_FOLLOWUP,
    BUILDER_SYSTEM_FALLBACK,
    get_builder_prompt,
)
from ..formatters import generate_build_summary
from .helpers import safe_send_event, store_message, stream_agent_events, NodeTimer


async def _execute_tool_calls(tool_calls, tool_map, event_queue) -> list[dict]:
    """Execute a list of tool calls and emit SSE events. Returns tool log."""
    tool_log: list[dict] = []
    results: dict[str, str] = {}

    for tc in tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]

        safe_send_event(event_queue, {
            "e": "tool_started",
            "tool_name": tool_name,
            "tool_input": tool_args,
        })

        matching = tool_map.get(tool_name)
        if matching:
            result = await matching.ainvoke(tool_args)
            output_str = str(result)[:500] if result else ""
            safe_send_event(event_queue, {
                "e": "tool_completed",
                "tool_name": tool_name,
                "tool_output": output_str[:150],
            })
            tool_log.append({"name": tool_name, "status": "success", "output": output_str[:150]})
            results[tc.get("id", tool_name)] = output_str
        else:
            print(f"Single-shot builder: unknown tool {tool_name}")

    return tool_log


async def _single_shot_build(
    builder_llm,
    tools: list,
    system_prompt: str,
    user_message: str,
    event_queue: asyncio.Queue,
    project_id: str,
    files_tracker: list,
) -> bool:
    """Single-shot builder with optional web search: up to 2 LLM calls.

    Pass 1: LLM may call web_search and/or write_multiple_files.
    Pass 2: If web_search was called but no files created, feed results back for a second call.

    Returns True if files were generated, False otherwise.
    """
    from langchain_core.messages import AIMessage, ToolMessage

    llm_with_tools = builder_llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}
    all_tool_log: list[dict] = []

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    safe_send_event(event_queue, {"e": "thinking", "message": "Generating all components..."})

    # Pass 1
    response = await llm_with_tools.ainvoke(messages)

    if not response.tool_calls:
        print("Single-shot builder: LLM returned no tool calls")
        return False

    tool_log = await _execute_tool_calls(response.tool_calls, tool_map, event_queue)
    all_tool_log.extend(tool_log)

    # Check if web_search was called but no files were created yet
    called_tools = {tc["name"] for tc in response.tool_calls}
    has_search = "web_search" in called_tools
    has_files = "write_multiple_files" in called_tools or "create_file" in called_tools

    if has_search and not has_files:
        # Pass 2: Feed search results back so LLM can generate code
        safe_send_event(event_queue, {"e": "thinking", "message": "Generating code with search results..."})

        # Build tool result messages for the conversation
        messages.append(response)  # AI message with tool calls
        for tc in response.tool_calls:
            tool_name = tc["name"]
            # Find the matching result from tool_log
            result_text = next(
                (entry["output"] for entry in tool_log if entry["name"] == tool_name),
                "Done",
            )
            messages.append(ToolMessage(content=result_text, tool_call_id=tc.get("id", tool_name)))

        response2 = await llm_with_tools.ainvoke(messages)

        if response2.tool_calls:
            tool_log2 = await _execute_tool_calls(response2.tool_calls, tool_map, event_queue)
            all_tool_log.extend(tool_log2)

    if all_tool_log:
        await store_message(
            chat_id=project_id,
            role="assistant",
            content=f"Executed {len(all_tool_log)} tool calls: {', '.join(t['name'] for t in all_tool_log)}",
            event_type="tool_summary",
            tool_calls=all_tool_log,
        )

    return len(files_tracker) > 0


async def builder_node(state: GraphState, config: RunnableConfig) -> dict:
    """Generate component and page code.

    First builds (scaffold ok): single-shot generation — 1 LLM call, all files at once.
    First builds (scaffold failed): ReAct agent with full tools for self-recovery.
    Follow-ups: ReAct agent with surgical edit tools.
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

        # ── Determine mode ──
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

        # Use the cheaper fast model for follow-up edits (e.g. Flash/Haiku)
        effective_model = get_fast_model(builder_model) if not is_first_message else builder_model
        builder_llm = create_llm(api_key, effective_model, max_tokens=16000)

        # Build the user message
        if not is_first_message:
            user_message = get_builder_prompt({"_user_prompt": user_prompt}, False)
        else:
            user_message = get_builder_prompt(plan, scaffold_ok)

        builder_error = None

        # ── Single-shot for first builds (scaffold ok) ──
        # 1 LLM call → all files at once. No ReAct loop.
        if is_first_message and scaffold_ok:
            try:
                success = await asyncio.wait_for(
                    _single_shot_build(
                        builder_llm, tools, system_prompt, user_message,
                        event_queue, project_id, files_tracker,
                    ),
                    timeout=120,
                )
                if not success:
                    builder_error = "Build failed — no files were generated. This may be due to an API issue. Please try again."
            except asyncio.TimeoutError:
                builder_error = "Build timed out. Please try a simpler prompt."
                print("Single-shot builder timed out after 2 minutes")
            except Exception as e:
                error_str = str(e)
                print(f"Single-shot builder error: {e}")
                traceback.print_exc()
                if "402" in error_str or "credits" in error_str.lower() or "afford" in error_str.lower():
                    builder_error = "Insufficient API credits. Please add credits to your API provider and try again."
                elif "429" in error_str or "rate" in error_str.lower():
                    builder_error = "API rate limit reached. Please wait a moment and try again."
                elif "401" in error_str or "unauthorized" in error_str.lower():
                    builder_error = "API authentication failed. Please check your API key."

        # ── ReAct agent for follow-ups and fallback ──
        else:
            agent_executor = create_react_agent(
                builder_llm,
                tools,
                prompt=SystemMessage(content=system_prompt),
            )
            messages = [HumanMessage(content=user_message)]

            try:
                await asyncio.wait_for(
                    stream_agent_events(agent_executor, messages, config, event_queue, project_id),
                    timeout=300,
                )
            except asyncio.TimeoutError:
                builder_error = "Build timed out after 5 minutes. Please try a simpler prompt."
                print("Builder agent timed out after 5 minutes")
            except Exception as e:
                error_str = str(e)
                print(f"Builder agent error: {e}")
                traceback.print_exc()
                if "402" in error_str or "credits" in error_str.lower() or "afford" in error_str.lower():
                    builder_error = "Insufficient API credits. Please add credits to your API provider and try again."
                elif "429" in error_str or "rate" in error_str.lower():
                    builder_error = "API rate limit reached. Please wait a moment and try again."
                elif "401" in error_str or "unauthorized" in error_str.lower():
                    builder_error = "API authentication failed. Please check your API key."

            # Detect empty builds on first message
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
