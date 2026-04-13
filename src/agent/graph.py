"""LangGraph graph — wires nodes together with conditional routing.

Flow:
    1. Router classifies the query and selects a model (MoM)
    2. Hevy node fetches workout data
    3. Volume node computes analysis + gaps
    4. Conditional: if query_type is "lookup", skip to synthesizer
    5. Otherwise: RAG retrieval + nutrition check run in parallel
    6. Synthesizer generates the final response using the routed model
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.agent.nodes import (
    hevy_node,
    nutrition_node,
    rag_node,
    router_node,
    synthesizer_node,
    volume_node,
)
from src.agent.state import AgentState


def should_skip_tools(state: AgentState) -> str:
    """After volume analysis, decide whether to skip RAG/nutrition.

    Lookups don't need literature retrieval or nutrition context —
    they just need the raw data formatted.
    """
    if state.get("query_type") == "lookup":
        return "synthesize"
    return "enrich"


def build_graph() -> StateGraph:
    """Build and compile the IronAgent graph."""

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("fetch_hevy", hevy_node)
    graph.add_node("analyze_volume", volume_node)
    graph.add_node("retrieve_literature", rag_node)
    graph.add_node("check_nutrition", nutrition_node)
    graph.add_node("synthesize", synthesizer_node)

    # Set entry point
    graph.set_entry_point("router")

    # Router → always fetch Hevy data
    graph.add_edge("router", "fetch_hevy")

    # Hevy → always analyze volume
    graph.add_edge("fetch_hevy", "analyze_volume")

    # Volume → conditional: skip tools for lookups, enrich for everything else
    graph.add_conditional_edges(
        "analyze_volume",
        should_skip_tools,
        {
            "synthesize": "synthesize",
            "enrich": "retrieve_literature",
        },
    )

    # RAG → nutrition (sequential since nutrition is fast)
    graph.add_edge("retrieve_literature", "check_nutrition")

    # Nutrition → synthesize
    graph.add_edge("check_nutrition", "synthesize")

    # Synthesize → end
    graph.add_edge("synthesize", END)

    return graph.compile()


# Pre-built graph instance
agent = build_graph()


def run_agent(query: str) -> dict:
    """Run a query through the full agent pipeline.

    Args:
        query: Natural language question about training.

    Returns:
        dict with "answer", "model_used", "query_type"
    """
    initial_state: AgentState = {
        "messages": [],
        "user_query": query,
        "query_type": "",
        "model": "",
        "hevy_data": None,
        "volume_analysis": None,
        "rag_context": None,
        "nutrition_constraints": None,
        "recommendation": None,
    }

    result = agent.invoke(initial_state)
    return result.get("recommendation", {"answer": "No response generated."})