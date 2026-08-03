from __future__ import annotations

import platform
import sys

import torch
from rich.console import Console
from rich.panel import Panel

from config import PROJECT_NAME, PROJECT_PATHS, load_configuration
from utils.logger import get_logger


def verify_project_folders() -> list[str]:
    """Return the configured project folders that exist on disk."""
    existing_paths: list[str] = []
    for label, path in PROJECT_PATHS.items():
        if path.exists():
            existing_paths.append(f"{label}: {path}")
    return existing_paths


def main() -> int:
    configuration = load_configuration()
    logger = get_logger(PROJECT_NAME)
    console = Console()

    console.print(Panel.fit(f"[bold cyan]{PROJECT_NAME}[/bold cyan]\nLocal-first intelligent support agent foundation"))
    logger.info("Configuration loaded for %s", configuration.project_name)

    existing_paths = verify_project_folders()
    system_info = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "loaded_directories": existing_paths,
    }

    console.print("[bold]System information[/bold]")
    for key, value in system_info.items():
        console.print(f"- {key}: {value}")

    logger.info("Project folders verified: %s", existing_paths)
    logger.info("Python %s | Torch %s | CUDA %s", system_info["python_version"], system_info["torch_version"], system_info["cuda_available"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
