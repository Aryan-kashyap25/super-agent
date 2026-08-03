from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is installed during setup.
    def load_dotenv() -> bool:  # type: ignore[no-redef]
        return False


PROJECT_NAME = "LocalMind AI"
PROJECT_DESCRIPTION = "Local-First Intelligent Support Agent powered by LangGraph"
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DEFAULT_DEVICE = "cuda" if os.getenv("FORCE_DEVICE") == "cuda" else "cpu"

FUTURE_MODEL_NAMES = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "generator_model": "localmind-placeholder-generator",
    "verifier_model": "localmind-placeholder-verifier",
}

PROJECT_PATHS = {
    "project_root": PROJECT_ROOT,
    "data_dir": DATA_DIR,
    "knowledge_base_dir": KNOWLEDGE_BASE_DIR,
    "model_dir": MODEL_DIR,
    "output_dir": OUTPUT_DIR,
    "graph_dir": PROJECT_ROOT / "graph",
    "nodes_dir": PROJECT_ROOT / "nodes",
    "retrieval_dir": PROJECT_ROOT / "retrieval",
    "llm_dir": PROJECT_ROOT / "llm",
    "verifier_dir": PROJECT_ROOT / "verifier",
    "utils_dir": PROJECT_ROOT / "utils",
    "tests_dir": PROJECT_ROOT / "tests",
}


@dataclass(frozen=True)
class SystemInfo:
    device: str
    python_executable: str
    python_version: str | None = None


@dataclass(frozen=True)
class Configuration:
    project_name: str
    project_description: str
    project_root: Path
    data_dir: Path
    knowledge_base_dir: Path
    model_dir: Path
    output_dir: Path
    log_level: str
    future_model_names: dict[str, str]
    device: str


def load_configuration() -> Configuration:
    load_dotenv()
    return Configuration(
        project_name=PROJECT_NAME,
        project_description=PROJECT_DESCRIPTION,
        project_root=PROJECT_ROOT,
        data_dir=DATA_DIR,
        knowledge_base_dir=KNOWLEDGE_BASE_DIR,
        model_dir=MODEL_DIR,
        output_dir=OUTPUT_DIR,
        log_level=LOG_LEVEL,
        future_model_names=FUTURE_MODEL_NAMES,
        device=get_device_name(),
    )


def get_device_name() -> str:
    if DEFAULT_DEVICE == "cuda":
        return "cuda"
    return "cpu"
