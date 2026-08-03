from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable, Literal

from retrieval.search import RetrievalEngine, SearchResult

from .state import AgentState


RouteName = Literal["answerable", "clarification", "escalation", "out_of_scope"]

SUPPORT_KEYWORDS = {
    "workspace",
    "dashboard",
    "connection",
    "connections",
    "export",
    "exports",
    "schedule",
    "api credential",
    "api credentials",
    "credential",
    "credentials",
    "role",
    "roles",
    "permission",
    "permissions",
    "timezone",
    "audit log",
    "audit logs",
    "resolved case",
    "support",
    "workspace settings",
    "refresh",
}

OUT_OF_SCOPE_KEYWORDS = {
    "weather",
    "joke",
    "movie",
    "music",
    "recipe",
    "capital",
    "sports",
    "news",
    "stock price",
    "translate",
}

ESCALATION_KEYWORDS = {
    "create for me",
    "please create",
    "change my role",
    "change roles",
    "delete data",
    "delete my",
    "contact recipient",
    "issue refund",
    "refund",
    "reset my password",
    "perform the export",
    "run the export",
    "make the change",
    "approve access",
}

VAGUE_PHRASES = {
    "help me",
    "something is wrong",
    "not working",
    "it broke",
    "this issue",
    "that issue",
    "i need help",
    "can you help",
}


def triage_node(state: AgentState) -> AgentState:
    question = _normalise_question(state.get("question", ""))
    classification, reason, confidence = classify_question(question)
    updates = {
        "classification": classification,
        "reason": reason,
        "confidence": confidence,
        "logs": _append_steps(state.get("logs"), "START", "TRIAGE", "ROUTING"),
        "execution_path": _append_steps(state.get("execution_path"), "START", "TRIAGE", "ROUTING"),
        "metadata": _merge_metadata(state.get("metadata"), {"triage": {"question": question, "reason": reason}}),
    }
    return updates


def clarification_node(state: AgentState) -> AgentState:
    message = "I need a little more information before I can answer your question."
    return {
        "answer": message,
        "requires_human": False,
        "logs": _append_steps(state.get("logs"), "CLARIFICATION", "END"),
        "execution_path": _append_steps(state.get("execution_path"), "CLARIFICATION", "END"),
        "metadata": _merge_metadata(state.get("metadata"), {"terminal_node": "clarification"}),
        "confidence": state.get("confidence", 0.0),
    }


def escalation_node(state: AgentState) -> AgentState:
    message = "This request requires human review or a supported operational workflow."
    return {
        "answer": message,
        "requires_human": True,
        "logs": _append_steps(state.get("logs"), "ESCALATION", "END"),
        "execution_path": _append_steps(state.get("execution_path"), "ESCALATION", "END"),
        "metadata": _merge_metadata(state.get("metadata"), {"terminal_node": "escalation"}),
        "confidence": state.get("confidence", 0.0),
    }


def out_of_scope_node(state: AgentState) -> AgentState:
    message = "This request is outside the supported knowledge base."
    return {
        "answer": message,
        "requires_human": False,
        "logs": _append_steps(state.get("logs"), "OUT_OF_SCOPE", "END"),
        "execution_path": _append_steps(state.get("execution_path"), "OUT_OF_SCOPE", "END"),
        "metadata": _merge_metadata(state.get("metadata"), {"terminal_node": "out_of_scope"}),
        "confidence": state.get("confidence", 0.0),
    }


def retrieval_node(state: AgentState, retrieval_engine: RetrievalEngine) -> AgentState:
    question = state.get("question", "")
    search_result = retrieval_engine.search(question)
    retrieved_docs = [_result_to_dict(item) for item in search_result["documents"]]
    retrieved_cases = [_result_to_dict(item) for item in search_result["resolved_cases"]]
    sources = retrieved_docs + retrieved_cases
    confidence = _derive_confidence(sources, default=state.get("confidence", 0.0))
    return {
        "retrieved_docs": retrieved_docs,
        "retrieved_cases": retrieved_cases,
        "sources": sources,
        "confidence": confidence,
        "logs": _append_steps(state.get("logs"), "RETRIEVAL"),
        "execution_path": _append_steps(state.get("execution_path"), "RETRIEVAL"),
        "metadata": _merge_metadata(
            state.get("metadata"),
            {
                "retrieval": {
                    "search_latency_seconds": search_result.get("search_latency_seconds", 0.0),
                    "index_size": search_result.get("index_size", 0),
                    "embedding_dimension": search_result.get("embedding_dimension", 0),
                }
            },
        ),
    }


def generation_node(state: AgentState, response_generator: Any) -> AgentState:
    question = state.get("question", "")
    retrieved_docs = state.get("retrieved_docs", [])
    retrieved_cases = state.get("retrieved_cases", [])
    generation_result = response_generator(question, retrieved_docs, retrieved_cases)
    return {
        "answer": generation_result.get("answer", ""),
        "generation_metadata": {
            "model_name": generation_result.get("model_name", ""),
            "generation_time": generation_result.get("generation_time", 0.0),
            "token_count": generation_result.get("token_count", 0),
            "prompt_used": generation_result.get("prompt_used", ""),
            "model_load_time": generation_result.get("model_load_time", 0.0),
            "prompt_creation_time": generation_result.get("prompt_creation_time", 0.0),
            "total_response_time": generation_result.get("total_response_time", 0.0),
        },
        "logs": _append_steps(state.get("logs"), "GENERATION", "END"),
        "execution_path": _append_steps(state.get("execution_path"), "GENERATION", "END"),
        "confidence": state.get("confidence", 0.0),
        "requires_human": False,
    }


def classify_question(question: str) -> tuple[RouteName, str, float]:
    if not question:
        return "clarification", "The question was empty or too vague to classify.", 0.0

    if _contains_keyword(question, OUT_OF_SCOPE_KEYWORDS):
        return "out_of_scope", "The request is unrelated to the supported OrbitDesk knowledge base.", 0.98

    if _contains_keyword(question, ESCALATION_KEYWORDS):
        return "escalation", "The request asks for an operational change or action that requires a human workflow.", 0.92

    if _is_vague(question):
        return "clarification", "The request needs more detail before a documented path can be chosen.", 0.52

    support_hits = _count_keyword_hits(question, SUPPORT_KEYWORDS)
    if support_hits > 0:
        return "answerable", "The question matches the supported OrbitDesk domain and can be answered from documentation.", 0.84

    if len(question.split()) <= 5:
        return "clarification", "The request is too short to choose a safe documented path.", 0.4

    return "out_of_scope", "No relevant OrbitDesk documentation terms were found in the request.", 0.75


def _normalise_question(question: str) -> str:
    return " ".join(question.strip().split())


def _contains_keyword(question: str, keywords: Iterable[str]) -> bool:
    lowered = question.lower()
    return any(keyword in lowered for keyword in keywords)


def _count_keyword_hits(question: str, keywords: Iterable[str]) -> int:
    lowered = question.lower()
    return sum(1 for keyword in keywords if keyword in lowered)


def _is_vague(question: str) -> bool:
    lowered = question.lower()
    if _contains_keyword(lowered, VAGUE_PHRASES):
        return True
    return len(lowered.split()) < 6 and not _contains_keyword(lowered, SUPPORT_KEYWORDS)


def _append_steps(existing: list[str] | None, *steps: str) -> list[str]:
    values = list(existing or [])
    values.extend(steps)
    return values


def _merge_metadata(existing: dict[str, Any] | None, updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing or {})
    merged.update(updates)
    return merged


def _result_to_dict(result: SearchResult) -> dict[str, Any]:
    if isinstance(result, dict):
        return dict(result)
    return asdict(result)


def _derive_confidence(sources: list[dict[str, Any]], default: float = 0.0) -> float:
    if not sources:
        return default
    scores = [float(item.get("similarity_score", 0.0)) for item in sources]
    return max(scores) if scores else default