from __future__ import annotations

from dataclasses import dataclass

import torch

from llm.generator import ResponseGenerator, generate_response
from llm.loader import LocalLLM
from llm.prompts import INSUFFICIENT_EVIDENCE_RESPONSE, SYSTEM_PROMPT, build_prompt


@dataclass
class FakeTokenizerOutput:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "</s>"
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, prompt, return_tensors="pt"):
        token_count = max(len(prompt.split()), 1)
        ids = torch.arange(1, token_count + 1, dtype=torch.long).unsqueeze(0)
        mask = torch.ones_like(ids)
        return {"input_ids": ids, "attention_mask": mask}

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(["The reset steps are described in the evidence."])


class FakeModel:
    def __init__(self):
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = torch.device(device)
        return self

    def eval(self):
        return self

    def generate(self, input_ids, attention_mask=None, max_new_tokens=256, temperature=0.2, top_p=0.9, do_sample=False, pad_token_id=None, eos_token_id=None):
        new_tokens = torch.tensor([[101, 102, 103]], dtype=torch.long)
        return torch.cat([input_ids, new_tokens], dim=1)


def test_model_loads_successfully(monkeypatch):
    def fake_tokenizer_loader(model_name, **kwargs):
        return FakeTokenizer()

    def fake_model_loader(model_name, **kwargs):
        if model_name == "preferred-model":
            raise OSError("preferred model unavailable")
        return FakeModel()

    monkeypatch.setattr("llm.loader.AutoTokenizer.from_pretrained", fake_tokenizer_loader)
    monkeypatch.setattr("llm.loader.AutoModelForCausalLM.from_pretrained", fake_model_loader)

    llm = LocalLLM.load(model_candidates=("preferred-model", "fallback-model"), device="cpu")

    assert llm.model_name == "fallback-model"
    assert llm.load_seconds >= 0


def test_prompt_generation_works():
    prompt = build_prompt(
        "How do I reset access?",
        [{"document_name": "KB-1", "chunk_text": "Reset the secret in settings.", "similarity_score": 0.99}],
        [{"document_name": "CASE-1", "chunk_text": "Rotate the secret and confirm revocation.", "similarity_score": 0.95}],
    )

    assert SYSTEM_PROMPT in prompt
    assert "How do I reset access?" in prompt
    assert "KB-1" in prompt
    assert "CASE-1" in prompt


def test_generator_returns_text(monkeypatch):
    monkeypatch.setattr("llm.loader.AutoTokenizer.from_pretrained", lambda *args, **kwargs: FakeTokenizer())
    monkeypatch.setattr("llm.loader.AutoModelForCausalLM.from_pretrained", lambda *args, **kwargs: FakeModel())
    llm = LocalLLM.load(model_candidates=("preferred-model", "fallback-model"), device="cpu")

    result = generate_response(
        "How do I reset access?",
        [{"document_name": "KB-1", "chunk_text": "Reset the secret in settings.", "similarity_score": 0.99}],
        [],
        llm=llm,
    )

    assert result["answer"]
    assert result["model_name"] == "preferred-model"
    assert result["token_count"] >= 0


def test_empty_evidence_handled_safely(monkeypatch):
    monkeypatch.setattr("llm.loader.AutoTokenizer.from_pretrained", lambda *args, **kwargs: FakeTokenizer())
    monkeypatch.setattr("llm.loader.AutoModelForCausalLM.from_pretrained", lambda *args, **kwargs: FakeModel())
    llm = LocalLLM.load(model_candidates=("preferred-model", "fallback-model"), device="cpu")

    result = generate_response("What should I do?", [], [], llm=llm)

    assert result["answer"] == INSUFFICIENT_EVIDENCE_RESPONSE
    assert result["token_count"] == 0


def test_response_metadata_returned_correctly(monkeypatch):
    monkeypatch.setattr("llm.loader.AutoTokenizer.from_pretrained", lambda *args, **kwargs: FakeTokenizer())
    monkeypatch.setattr("llm.loader.AutoModelForCausalLM.from_pretrained", lambda *args, **kwargs: FakeModel())
    llm = LocalLLM.load(model_candidates=("preferred-model", "fallback-model"), device="cpu")

    result = ResponseGenerator.create(llm=llm).generate(
        "How do I reset access?",
        [{"document_name": "KB-1", "chunk_text": "Reset the secret in settings.", "similarity_score": 0.99}],
        [],
    )

    assert result["model_name"] == "preferred-model"
    assert "prompt_used" in result
    assert result["generation_time"] >= 0
    assert result["prompt_creation_time"] >= 0
    assert result["total_response_time"] >= 0