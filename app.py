from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from rich import box
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.table import Table

from config import PROJECT_NAME, PROJECT_PATHS, load_configuration
from graph.graph import GraphDependencies, build_support_graph
from llm.generator import generate_response
from llm.loader import load_local_llm
from retrieval.search import build_retrieval_engine
from utils.logger import get_logger
from verifier.verifier import SAFE_FAILURE_MESSAGE, verify_generated_response


APP_VERSION = "1.0.0"


@dataclass
class LazyRetrievalEngine:
    engine: Any | None = None
    load_seconds: float = 0.0
    error: str | None = None

    def search(self, question: str) -> dict[str, Any]:
        if self.engine is None and self.error is None:
            load_start = time.perf_counter()
            try:
                self.engine = build_retrieval_engine()
                self.load_seconds = self.engine.document_load_seconds + self.engine.chunk_seconds + self.engine.embedding_seconds + self.engine.index_seconds
            except Exception as exc:  # pragma: no cover - runtime fallback path.
                self.error = str(exc)
                self.load_seconds = time.perf_counter() - load_start
                return {
                    "documents": [],
                    "resolved_cases": [],
                    "search_latency_seconds": 0.0,
                    "index_size": 0,
                    "embedding_dimension": 0,
                    "error": self.error,
                }

        if self.engine is None:
            return {
                "documents": [],
                "resolved_cases": [],
                "search_latency_seconds": 0.0,
                "index_size": 0,
                "embedding_dimension": 0,
                "error": self.error,
            }

        result = self.engine.search(question)
        result["embedding_load_time"] = self.load_seconds
        return result


@dataclass
class RuntimeContext:
    console: Console = field(default_factory=Console)
    logger_name: str = PROJECT_NAME
    graph: Any | None = None
    retrieval_provider: LazyRetrievalEngine = field(default_factory=LazyRetrievalEngine)
    llm: Any | None = None


def build_runtime_context() -> RuntimeContext:
    context = RuntimeContext()
    context.llm = None

    def _lazy_response_generator(question: str, retrieved_documents: list[dict[str, Any]], retrieved_cases: list[dict[str, Any]], llm: Any | None = None) -> dict[str, Any]:
        active_llm = llm or context.llm
        if active_llm is None:
            try:
                active_llm = load_local_llm()
                context.llm = active_llm
            except Exception as exc:  # pragma: no cover - runtime fallback path.
                return {
                    "answer": SAFE_FAILURE_MESSAGE,
                    "model_name": "unavailable",
                    "generation_time": 0.0,
                    "token_count": 0,
                    "prompt_used": f"Model unavailable: {exc}",
                    "model_load_time": 0.0,
                    "prompt_creation_time": 0.0,
                    "total_response_time": 0.0,
                }

        try:
            return generate_response(question, retrieved_documents, retrieved_cases, llm=active_llm)
        except Exception as exc:  # pragma: no cover - runtime fallback path.
            return {
                "answer": SAFE_FAILURE_MESSAGE,
                "model_name": getattr(active_llm, "model_name", "unavailable"),
                "generation_time": 0.0,
                "token_count": 0,
                "prompt_used": f"Generation unavailable: {exc}",
                "model_load_time": getattr(active_llm, "load_seconds", 0.0),
                "prompt_creation_time": 0.0,
                "total_response_time": 0.0,
            }

    dependencies = GraphDependencies(
        retrieval_engine=context.retrieval_provider,
        response_generator=_lazy_response_generator,
        verification_engine=verify_generated_response,
        llm=None,
    )
    context.graph = build_support_graph(dependencies)
    return context


def verify_project_folders() -> list[str]:
    """Return the configured project folders that exist on disk."""
    existing_paths: list[str] = []
    for label, path in PROJECT_PATHS.items():
        if path.exists():
            existing_paths.append(f"{label}: {path}")
    return existing_paths


def format_execution_trace(execution_path: list[str]) -> str:
    if not execution_path:
        return "START\n↓\nEND"
    return "\n↓\n".join(execution_path)


def format_sources_table(sources: list[dict[str, Any]]) -> Table:
    table = Table(title="Retrieved Sources", box=box.SIMPLE_HEAVY)
    table.add_column("Document", style="cyan", no_wrap=True)
    table.add_column("Chunk / Passage", style="magenta")
    table.add_column("Score", justify="right")
    table.add_column("Type", style="green")
    if not sources:
        table.add_row("-", "-", "-", "-")
        return table

    for item in sources:
        table.add_row(
            str(item.get("document") or item.get("document_name") or "-"),
            str(item.get("chunk_id") or item.get("passage_id") or "-"),
            f"{float(item.get('similarity_score', 0.0) or 0.0):.2f}",
            str(item.get("source_type", "knowledge_base")),
        )
    return table


def print_banner(console: Console) -> None:
    console.print(Panel.fit(f"[bold cyan]{PROJECT_NAME}[/bold cyan]\nLocalMind AI - Version {APP_VERSION}", border_style="cyan"))


def run_question(graph: Any, question: str) -> dict[str, Any]:
    return graph.invoke({"question": question, "logs": [], "execution_path": [], "retry_count": 0, "metadata": {}})


def print_response(console: Console, state: dict[str, Any]) -> None:
    console.print(Panel.fit(f"[bold]Final Answer[/bold]\n{state.get('answer', '')}", border_style="green"))
    console.print("[bold]Execution Trace[/bold]")
    console.print(format_execution_trace(list(state.get("execution_path", []))))
    console.print(format_sources_table(list(state.get("sources", []))))

    metrics = {
        "model_load_time": state.get("generation_metadata", {}).get("model_load_time", 0.0),
        "embedding_load_time": state.get("metadata", {}).get("retrieval", {}).get("embedding_load_time", 0.0),
        "retrieval_latency": state.get("metadata", {}).get("retrieval", {}).get("search_latency_seconds", 0.0),
        "generation_latency": state.get("generation_metadata", {}).get("generation_time", 0.0),
        "verification_latency": state.get("verification_metadata", {}).get("verification_time", 0.0),
        "total_response_time": state.get("generation_metadata", {}).get("total_response_time", 0.0),
    }
    metrics_table = Table(title="Performance Metrics", box=box.SIMPLE_HEAVY)
    metrics_table.add_column("Metric")
    metrics_table.add_column("Seconds", justify="right")
    for key, value in metrics.items():
        metrics_table.add_row(key, f"{float(value or 0.0):.4f}")
    console.print(metrics_table)

    console.print(Panel.fit(f"[bold]Structured JSON[/bold]", border_style="blue"))
    console.print(JSON.from_data(state))


def main() -> int:
    configuration = load_configuration()
    logger = get_logger(PROJECT_NAME)
    console = Console()
    runtime = build_runtime_context()

    print_banner(console)
    logger.info("Configuration loaded for %s", configuration.project_name)

    existing_paths = verify_project_folders()
    graph_status = "compiled"
    system_info = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "graph_status": graph_status,
        "llm_model_candidates": configuration.llm_model_candidates,
        "loaded_directories": existing_paths,
    }

    console.print("[bold]Model information[/bold]")
    console.print(f"- embedding model: {configuration.future_model_names['embedding_model']}")
    console.print(f"- generator models: {configuration.llm_model_candidates}")
    console.print(f"- device: {system_info['device']}")

    console.print("[bold]Graph status[/bold]")
    console.print(f"- status: {graph_status}")
    console.print(f"- python version: {system_info['python_version']}")
    console.print(f"- torch version: {system_info['torch_version']}")
    console.print(f"- cuda available: {system_info['cuda_available']}")
    console.print("[bold]Loaded directories[/bold]")
    for entry in existing_paths:
        console.print(f"- {entry}")

    console.print("[bold]Commands[/bold]: exit | quit | clear")

    if runtime.graph is None:
        console.print("[red]Graph initialization failed.[/red]")
        return 1

    while True:
        try:
            question = console.input("[bold cyan]Question[/bold cyan] > ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nExiting LocalMind AI.")
            break

        if not question:
            console.print("[yellow]Please enter a question or type exit.[/yellow]")
            continue

        lowered = question.lower()
        if lowered in {"exit", "quit"}:
            console.print("Exiting LocalMind AI.")
            break
        if lowered == "clear":
            console.clear()
            print_banner(console)
            continue

        start_time = time.perf_counter()
        try:
            state = run_question(runtime.graph, question)
        except Exception as exc:  # pragma: no cover - runtime fallback path.
            logger.exception("Graph execution failed for question: %s", question)
            console.print(Panel.fit(f"[red]Execution failed:[/red] {exc}", border_style="red"))
            continue

        state.setdefault("generation_metadata", {})
        state.setdefault("metadata", {})
        state.setdefault("verification_metadata", {})
        state["generation_metadata"].setdefault("total_response_time", time.perf_counter() - start_time)
        if "retrieval" in state["metadata"] and "embedding_load_time" not in state["metadata"]["retrieval"]:
            state["metadata"]["retrieval"]["embedding_load_time"] = runtime.retrieval_provider.load_seconds

        logger.info(
            "Question=%s | classification=%s | retry=%s | execution_time=%.4fs",
            question,
            state.get("classification"),
            state.get("retry_count", 0),
            state["generation_metadata"].get("total_response_time", 0.0),
        )
        print_response(console, state)

    logger.info("Project folders verified: %s", existing_paths)
    logger.info("Python %s | Torch %s | CUDA %s", system_info["python_version"], system_info["torch_version"], system_info["cuda_available"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
