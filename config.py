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

    # Primary lightweight model
    "generator_model": "Qwen/Qwen2.5-0.5B-Instruct",

    # Fallback
    "generator_fallback_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",

    "verifier_model": "localmind-placeholder-verifier",
}

LLM_MODEL_CANDIDATES = (
    FUTURE_MODEL_NAMES["generator_model"],
    FUTURE_MODEL_NAMES["generator_fallback_model"],
)


@dataclass(frozen=True)
class GenerationSettings:
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool


LLM_GENERATION_SETTINGS = GenerationSettings(
    max_new_tokens=int(os.getenv("LLM_MAX_NEW_TOKENS", "256")),
    temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
    top_p=float(os.getenv("LLM_TOP_P", "0.9")),
    do_sample=os.getenv("LLM_DO_SAMPLE", "false").strip().lower() == "true",
)

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
    llm_model_candidates: tuple[str, str]
    llm_generation_settings: GenerationSettings
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
        llm_model_candidates=LLM_MODEL_CANDIDATES,
        llm_generation_settings=LLM_GENERATION_SETTINGS,
        device=get_device_name(),
    )


def get_device_name() -> str:
    if DEFAULT_DEVICE == "cuda":
        return "cuda"
    return "cpu"
