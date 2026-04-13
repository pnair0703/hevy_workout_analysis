"""Agent state — the TypedDict that flows through the LangGraph."""

from __future__ import annotations

from typing import Annotated, Any, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """State that flows through every node in the graph.

    messages: conversation history (LangGraph manages appending)
    user_query: the raw user question
    query_type: classification from the router (lookup / programming / coaching / research)
    model: which LLM to route to based on query_type
    hevy_data: workout data pulled from the Hevy API
    volume_analysis: computed volume stats from the calculator
    rag_context: retrieved sports science text
    nutrition_constraints: phase-adjusted training guidance
    recommendation: the final structured output
    """

    messages: Annotated[list, add_messages]
    user_query: str
    query_type: str
    model: str
    hevy_data: Optional[dict[str, Any]]
    volume_analysis: Optional[dict[str, Any]]
    rag_context: Optional[str]
    nutrition_constraints: Optional[dict[str, Any]]
    recommendation: Optional[dict[str, Any]]