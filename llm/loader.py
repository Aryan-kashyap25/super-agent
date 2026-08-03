from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_DIR, get_device_name, load_configuration
from utils.helpers import ensure_directory
from utils.logger import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class LoadedModelArtifacts:
    model_name: str
    tokenizer: Any
    model: Any
    device: str
    dtype: torch.dtype


@dataclass
class LocalLLM:
    """Reusable local language model wrapper with cached Hugging Face artifacts."""

    model_name: str
    tokenizer: Any
    model: Any
    device: str
    dtype: torch.dtype
    cache_dir: Path
    load_seconds: float
    _generation_context: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        model_candidates: tuple[str, ...] | None = None,
        cache_dir: Path | None = None,
        device: str | None = None,
    ) -> "LocalLLM":
        configuration = load_configuration()
        cache_directory = ensure_directory(cache_dir or MODEL_DIR)
        selected_device = device or get_device_name()
        candidates = model_candidates or configuration.llm_model_candidates
        load_start = time.perf_counter()

        last_error: Exception | None = None
        for model_name in candidates:
            try:
                artifacts = _load_model_artifacts(model_name, selected_device, cache_directory)
                load_seconds = time.perf_counter() - load_start
                logger.info("Model loading: %s on %s in %.2fs", artifacts.model_name, artifacts.device, load_seconds)
                logger.info("Device: %s | dtype: %s", artifacts.device, artifacts.dtype)
                return cls(
                    model_name=artifacts.model_name,
                    tokenizer=artifacts.tokenizer,
                    model=artifacts.model,
                    device=artifacts.device,
                    dtype=artifacts.dtype,
                    cache_dir=cache_directory,
                    load_seconds=load_seconds,
                )
            except Exception as exc:  # pragma: no cover - fallback path exercised in tests.
                last_error = exc
                logger.warning("Failed to load model %s: %s", model_name, exc)

        raise RuntimeError("Unable to load any configured local language model.") from last_error

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ) -> tuple[str, int, float]:
        generation_start = time.perf_counter()
        tokenized = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(self.model.device if hasattr(self.model, "device") else self.device)
        attention_mask = tokenized.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(input_ids.device)

        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
            eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
        )
        generation_seconds = time.perf_counter() - generation_start

        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt) :]
        cleaned_text = generated_text.strip()
        token_count = max(int(output_ids.shape[-1] - input_ids.shape[-1]), 0)
        return cleaned_text, token_count, generation_seconds


@lru_cache(maxsize=4)
def _load_model_artifacts(model_name: str, device: str, cache_dir: Path) -> LoadedModelArtifacts:
    dtype = _select_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir))
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=str(cache_dir), torch_dtype=dtype)

    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()

    return LoadedModelArtifacts(model_name=model_name, tokenizer=tokenizer, model=model, device=device, dtype=dtype)


def _select_dtype(device: str) -> torch.dtype:
    if device == "cuda" and torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_local_llm(
    model_candidates: tuple[str, ...] | None = None,
    cache_dir: Path | None = None,
    device: str | None = None,
) -> LocalLLM:
    return LocalLLM.load(model_candidates=model_candidates, cache_dir=cache_dir, device=device)