"""
LangGraph workflow builder for the orchestration.

First build:  planner → scaffold → builder → app_start ⇄ fixer → END
Follow-up:    builder → build_checkpoint ⇄ fixer → app_start ⇄ fixer → END

First builds skip build_checkpoint for speed (single-shot builder + direct app start).
Follow-ups still validate with vite build since surgical edits are more error-prone.
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


def route_entry(state: GraphState) -> str:
    """Route entry: first build goes through planner, follow-ups skip to builder."""
    if state.get("is_first_message", True):
        return "planner"
    print("Follow-up — skipping planner/scaffold, going straight to builder")
    return "builder"


def route_after_builder(state: GraphState) -> str:
    """Route after builder: first builds skip validation, follow-ups validate.

    First builds use single-shot generation and go straight to app_start.
    Follow-ups go through build_checkpoint for vite build validation.
    """
    if state.get("error_message"):
        print("Builder had an error — going to app_start")
        return "app_start"
    if state.get("is_first_message", True):
        print("First build — skipping build_checkpoint, going straight to app_start")
        return "app_start"
    print("Follow-up — running build_checkpoint for validation")
    return "build_checkpoint"


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
    """Build the LangGraph state machine.

    First build:
      planner → scaffold → builder ────────────────────→ app_start ─── (ok) ───→ END
                                                              │
                                                              └── fixer → build_checkpoint → ...

    Follow-up:
      builder → build_checkpoint ─── (pass) ───→ app_start ─── (ok) ───→ END
                      ↑                  │           │
                      └── fixer ◄────────┘───────────┘
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("router", lambda state: state)  # passthrough, routing handled by conditional edges
    workflow.add_node("planner", planner_node)
    workflow.add_node("scaffold", scaffold_node)
    workflow.add_node("builder", builder_node)
    workflow.add_node("build_checkpoint", build_checkpoint_node)
    workflow.add_node("fixer", fixer_node)
    workflow.add_node("app_start", app_start_node)

    # Entry: router decides first-build vs follow-up
    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router",
        route_entry,
        {"planner": "planner", "builder": "builder"},
    )

    # First-build path: planner → scaffold → builder
    workflow.add_edge("planner", "scaffold")
    workflow.add_edge("scaffold", "builder")

    # After builder: first builds skip to app_start, follow-ups validate
    workflow.add_conditional_edges(
        "builder",
        route_after_builder,
        {"app_start": "app_start", "build_checkpoint": "build_checkpoint"},
    )

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
