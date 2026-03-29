from langgraph.constants import START, END
from langgraph.graph import StateGraph

from app.core.state import AgentActionState
from nodes import classify_intent, classify_sources, data_agent, analysis_agent, report_agent, draft_response


def build_graph():
    builder = StateGraph(AgentActionState)
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("classify_sources", classify_sources)
    builder.add_node("data_agent", data_agent)
    builder.add_node("analysis_agent", analysis_agent)
    builder.add_node("report_agent", report_agent)
    builder.add_node("draft_response", draft_response)

    builder.add_edge(START, "classify_intent")
    builder.add_edge("draft_response", END)

    return builder.compile()


app = build_graph()