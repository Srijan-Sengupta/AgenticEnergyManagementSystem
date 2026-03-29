from typing import TypedDict, Literal, Optional, List, Dict, Any


class ActionClassification(TypedDict):
    intent: Literal["data_add", "data_modify", "query", "report", "irrelevant"]
    summary: str
    reasoning: str


class SourceClassification(TypedDict):
    source: Literal["outage_reports", "demand_reports", "both"]
    reasoning: str


class AgentActionState(TypedDict):
    user_request: str
    classification: Optional[ActionClassification]
    source_classification: Optional[SourceClassification]

    issue_search_results: Optional[List[str]]
    db_search_results: Optional[Dict[str, Any]]

    messages: List[str]
    drafted_response: Optional[str]