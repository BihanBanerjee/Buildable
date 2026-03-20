"""
LangGraph workflow builder for the 6-node orchestration.

Pipeline: planner → scaffold → builder → build_checkpoint ⇄ fixer → app_start ⇄ fixer → END

- planner: Fast model generates the plan (structured output)
- scaffold: Deterministic — installs deps, generates App.jsx with routes, zero LLM
- builder: Expensive model creates component/page files via ReAct agent
- build_checkpoint: Deterministic — runs npx vite build, zero LLM
- fixer: Cheap model (Flash) fixes build errors, max 2 retries
- app_start: Deterministic — ensures Vite dev server is running + runtime error check
"""

from langgraph.graph import StateGraph, END
from .graph_state import GraphState
from .nodes import (
    planner_node,
    scaffold_node,
    builder_node,
    build_checkpoint_node,
    fixer_node,
    app_start_node,
)


def should_fix_or_start(state: GraphState) -> str:
    """Route after build_checkpoint: fix errors or start the app."""
    if state.get("build_passed"):
        print("Build checkpoint PASSED — starting app")
        return "app_start"

    retries = state.get("fixer_retries", 0)
    max_retries = state.get("max_fixer_retries", 2)

    if retries < max_retries:
        print(f"Build checkpoint FAILED — sending to fixer (attempt {retries + 1}/{max_retries})")
        return "fixer"

    print(f"Fixer exhausted ({retries}/{max_retries} retries) — starting app anyway")
    return "app_start"


def should_end_or_fix_runtime(state: GraphState) -> str:
    """Route after app_start: end if success, or send runtime errors to fixer."""
    if state.get("success"):
        print("App start PASSED — workflow complete")
        return "end"

    retries = state.get("fixer_retries", 0)
    max_retries = state.get("max_fixer_retries", 2)

    if retries < max_retries:
        error_msg = state.get("error_message", "")
        print(f"App start detected runtime errors — sending to fixer (attempt {retries + 1}/{max_retries})")
        print(f"Runtime errors: {error_msg[:200]}")
        return "fixer"

    print(f"Fixer exhausted ({retries}/{max_retries} retries) — ending anyway")
    return "end"


def create_langgraph_workflow():
    """Build the 6-node LangGraph state machine.

    planner → scaffold → builder → build_checkpoint ─── (pass) ───→ app_start ─── (ok) ───→ END
                                          ↑                  │           │
                                          │            (fail, retries)   │ (runtime errors, retries left)
                                          │                  │           │
                                          └── fixer ◄────────┘───────────┘
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("scaffold", scaffold_node)
    workflow.add_node("builder", builder_node)
    workflow.add_node("build_checkpoint", build_checkpoint_node)
    workflow.add_node("fixer", fixer_node)
    workflow.add_node("app_start", app_start_node)

    # Linear: planner → scaffold → builder → build_checkpoint
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "scaffold")
    workflow.add_edge("scaffold", "builder")
    workflow.add_edge("builder", "build_checkpoint")

    # Conditional: build_checkpoint → fixer or app_start
    workflow.add_conditional_edges(
        "build_checkpoint",
        should_fix_or_start,
        {"fixer": "fixer", "app_start": "app_start"},
    )

    # Fixer loops back to build_checkpoint
    workflow.add_edge("fixer", "build_checkpoint")

    # Conditional: app_start → end or fixer (for runtime errors)
    workflow.add_conditional_edges(
        "app_start",
        should_end_or_fix_runtime,
        {"end": END, "fixer": "fixer"},
    )

    return workflow.compile()


_app = create_langgraph_workflow()


def get_workflow():
    """Return the pre-compiled workflow."""
    return _app
