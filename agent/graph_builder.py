from langgraph.graph import StateGraph, END
from .graph_state import GraphState
from .graph_nodes import (
    planner_builder_node,
    build_checkpoint_node,
    fixer_node,
    app_start_node,
)


def should_fix_or_start(state: GraphState) -> str:
    """Route after build checkpoint: fix errors or start app."""
    if state.get("build_passed", False):
        print("Build checkpoint PASSED — skipping fixer, starting app")
        return "app_start"

    retries = state.get("fixer_retries", 0)
    max_retries = state.get("max_fixer_retries", 2)

    if retries < max_retries:
        print(f"Build checkpoint FAILED — sending to fixer (attempt {retries + 1}/{max_retries})")
        return "fixer"
    else:
        print(f"Fixer exhausted ({retries}/{max_retries} retries) — starting app anyway")
        return "app_start"


def create_langgraph_workflow():
    """Build the 2-agent + deterministic checkpoint graph.

    planner_builder → build_checkpoint ─── (pass) ───→ app_start → END
                            ↑               │
                            │          (fail, retries left)
                            │               │
                            └── fixer ◄─────┘
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("planner_builder", planner_builder_node)
    workflow.add_node("build_checkpoint", build_checkpoint_node)
    workflow.add_node("fixer", fixer_node)
    workflow.add_node("app_start", app_start_node)

    workflow.set_entry_point("planner_builder")
    workflow.add_edge("planner_builder", "build_checkpoint")

    workflow.add_conditional_edges(
        "build_checkpoint",
        should_fix_or_start,
        {"fixer": "fixer", "app_start": "app_start"},
    )

    # After fixer patches files, re-run the build checkpoint
    workflow.add_edge("fixer", "build_checkpoint")

    workflow.add_edge("app_start", END)

    return workflow.compile()


_app = create_langgraph_workflow()


def get_workflow():
    """Return the compiled LangGraph workflow."""
    return _app
