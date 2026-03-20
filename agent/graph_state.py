from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator


class GraphState(TypedDict):
    """State schema for the 2-agent + deterministic checkpoint orchestration.

    Pipeline: planner_builder → build_checkpoint ⇄ fixer → app_start
    """

    project_id: str
    user_prompt: str
    is_first_message: bool

    # Plan (produced by planner_builder, used for description/summary)
    plan: Optional[Dict[str, Any]]

    # Files tracking (accumulates across fixer retries)
    files_created: Annotated[List[str], operator.add]

    # Build checkpoint results
    build_passed: bool
    build_errors: str  # stderr from failed vite build (empty if passed)

    # Fixer retry tracking
    fixer_retries: int
    max_fixer_retries: int

    # Node tracking
    current_node: str
    execution_log: Annotated[List[Dict[str, Any]], operator.add]

    # Model selection (full OpenRouter model ID, e.g. "google/gemini-2.5-pro")
    builder_model: str

    # Results
    success: bool
    error_message: Optional[str]
