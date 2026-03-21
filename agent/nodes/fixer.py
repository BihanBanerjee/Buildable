"""
Node 5: FIXER — cheap model fixes build errors (Gemini Flash only).

Tools: read_file, create_file, execute_command (restricted to npm install).
Always uses Flash regardless of builder model choice.
"""

import asyncio
import traceback

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent

from ..graph_state import GraphState
from ..agent import create_llm
from ..tools import create_tools
from ..prompts import FIXER_PROMPT, get_fixer_prompt
from .helpers import stream_agent_events, NodeTimer


async def fixer_node(state: GraphState, config: RunnableConfig) -> dict:
    """Fix build errors using Gemini Flash.

    Tools: read_file, create_file, execute_command (restricted to npm install).
    Always uses Flash regardless of builder model — fast, cheap, follows instructions.
    """
    timer = NodeTimer("fixer")
    configurable = config.get("configurable", {})
    event_queue = configurable.get("event_queue")
    sandbox = configurable.get("sandbox")

    fixer_retries = state.get("fixer_retries", 0)

    try:
        if not sandbox:
            raise Exception("Sandbox not available")

        build_errors = state.get("build_errors", "Unknown error")
        project_id = state.get("project_id", "")
        plan = state.get("plan", {})
        api_key = configurable.get("openrouter_api_key")
        fast_model = "google/gemini-2.5-flash"

        # Determine error type: if build passed but we're in fixer, it's a runtime error
        build_passed = state.get("build_passed", False)
        error_type = "runtime" if build_passed else "build"
        print(f"Fixer node: attempt {fixer_retries + 1} ({error_type} error), fixing:\n{build_errors[:300]}")

        tools = create_tools(sandbox, event_queue, project_id, mode="fixer")
        fixer_llm = create_llm(api_key, fast_model, max_tokens=4000)

        agent_executor = create_react_agent(
            fixer_llm,
            tools,
            prompt=SystemMessage(content=FIXER_PROMPT),
        )

        user_message = get_fixer_prompt(build_errors, plan=plan, error_type=error_type)
        messages = [HumanMessage(content=user_message)]

        try:
            await asyncio.wait_for(
                stream_agent_events(agent_executor, messages, config, event_queue, project_id),
                timeout=120,
            )
        except asyncio.TimeoutError:
            print("Fixer agent timed out after 2 minutes")
        except Exception as e:
            print(f"Fixer agent error: {e}")
            traceback.print_exc()

        log_entry = timer.stop(attempt=fixer_retries + 1)
        return {
            "fixer_retries": fixer_retries + 1,
            "current_node": "fixer",
            "execution_log": [log_entry],
        }

    except Exception as e:
        error_msg = f"Fixer node error: {str(e)}"
        print(error_msg)

        log_entry = timer.stop(status="error", error=error_msg, attempt=fixer_retries + 1)
        return {
            "fixer_retries": fixer_retries + 1,
            "current_node": "fixer",
            "error_message": error_msg,
            "execution_log": [log_entry],
        }
