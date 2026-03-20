"""
Node 1: PLANNER — fast model generates the application plan.
"""

import asyncio
import json
import traceback

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from ..graph_state import GraphState
from ..agent import create_llm, get_fast_model
from ..prompts import ENHANCER_PLANNER_PROMPT
from ..formatters import create_formatted_message, generate_description_from_plan
from .helpers import PlanSchema, safe_send_event, store_message, NodeTimer
from utils.store import load_json_store


async def planner_node(state: GraphState, config: RunnableConfig) -> dict:
    """Generate the application plan using a fast model.

    Emits: enhancer_started, planner_started, planner_complete, description.
    """
    timer = NodeTimer("planner")
    configurable = config.get("configurable", {})
    event_queue = configurable.get("event_queue")

    try:
        user_prompt = state.get("user_prompt", "")
        project_id = state.get("project_id", "")
        is_first_message = state.get("is_first_message", True)
        api_key = configurable.get("openrouter_api_key")
        builder_model = state.get("builder_model", "google/gemini-2.5-pro")
        fast_model = get_fast_model(builder_model)

        safe_send_event(event_queue, {"e": "enhancer_started", "message": "Understanding your idea..."})
        safe_send_event(event_queue, {"e": "planner_started", "message": "Planning the application architecture..."})

        # Load previous context for follow-ups
        previous_context = ""
        if project_id and not is_first_message:
            context = await asyncio.to_thread(load_json_store, project_id, "context.json")
            if context:
                conv_text = ""
                for i, conv in enumerate(context.get("conversation_history", [])[-5:], 1):
                    status = "[OK]" if conv.get("success") else "[FAIL]"
                    conv_text += f"  {i}. {status} {conv.get('user_prompt', '')[:80]}\n"

                previous_context = (
                    f"\n\nPREVIOUS PROJECT CONTEXT:"
                    f"\n{context.get('semantic', '')}"
                    f"\nFiles: {len(context.get('files_created', []))} created"
                    f"\nHistory:\n{conv_text}"
                )

        planner_llm = create_llm(api_key, fast_model, max_tokens=4000)

        plan_messages = [
            SystemMessage(content=ENHANCER_PLANNER_PROMPT),
            HumanMessage(content=f"{user_prompt}{previous_context}"),
        ]

        try:
            structured_llm = planner_llm.with_structured_output(PlanSchema)
            plan_response = await structured_llm.ainvoke(plan_messages)
            plan = plan_response.dict() if hasattr(plan_response, "dict") else plan_response.model_dump()
        except Exception:
            # Fallback: raw text → parse as JSON
            raw_response = await planner_llm.ainvoke(plan_messages)
            raw_text = raw_response.content.strip()
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
            try:
                plan = json.loads(raw_text)
            except json.JSONDecodeError:
                plan = {
                    "overview": user_prompt,
                    "components": [],
                    "pages": [],
                    "dependencies": [],
                    "file_structure": [],
                    "implementation_steps": [f"Build the application: {user_prompt}"],
                }

        print(f"Plan generated: {json.dumps(plan, indent=2)[:500]}")

        # Emit plan events for frontend
        formatted_plan = create_formatted_message("planner_complete", plan)
        safe_send_event(event_queue, formatted_plan)

        description = generate_description_from_plan(plan, user_prompt)
        safe_send_event(event_queue, {"e": "description", "message": description})

        await store_message(chat_id=project_id, role="assistant", content=description, event_type="description")

        log_entry = timer.stop()
        return {
            "plan": plan,
            "current_node": "planner",
            "execution_log": [log_entry],
        }

    except Exception as e:
        error_msg = f"Planner error: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        if event_queue:
            safe_send_event(event_queue, {"e": "planner_error", "message": error_msg})

        log_entry = timer.stop(status="error", error=error_msg)
        return {
            "plan": {"overview": state.get("user_prompt", ""), "components": [], "pages": ["Home"], "dependencies": [], "file_structure": [], "implementation_steps": []},
            "current_node": "planner",
            "error_message": error_msg,
            "execution_log": [log_entry],
        }
