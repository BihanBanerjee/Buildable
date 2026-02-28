from langgraph.graph import StateGraph, END
from .graph_state import GraphState
from .graph_nodes import (
    planner_node,
    builder_node,
    code_validator_node,
    application_checker_node,
    should_retry_builder_for_validation,
    should_retry_builder_or_finish,
)


def create_langgraph_workflow():
    """Returns the LangGraph workflow for the multi-agent system"""

    # Create the state graph
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("builder", builder_node)
    workflow.add_node("code_validator", code_validator_node)
    workflow.add_node("application_checker", application_checker_node)

    # Set entry point
    workflow.set_entry_point("planner")

    # Add edges with retry logic
    workflow.add_edge("planner", "builder")
    workflow.add_edge("builder", "code_validator")

    # Code validator can retry builder or continue to app checker
    workflow.add_conditional_edges(
        "code_validator",
        should_retry_builder_for_validation,
        {"builder": "builder", "application_checker": "application_checker"},
    )

    # Application checker can retry builder or finish
    workflow.add_conditional_edges(
        "application_checker",
        should_retry_builder_or_finish,
        {"builder": "builder", "end": END},
    )

    # Compile the workflow
    return workflow.compile()


_app = create_langgraph_workflow()


def get_workflow():
    """Return the compiled LangGraph workflow."""
    return _app
