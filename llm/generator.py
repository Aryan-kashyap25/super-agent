from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from config import load_configuration
from utils.logger import get_logger

from .loader import LocalLLM, load_local_llm
from .prompts import INSUFFICIENT_EVIDENCE_RESPONSE, build_prompt


logger = get_logger(__name__)


@dataclass(frozen=True)
class GenerationResponse:
    answer: str
    model_name: str
    generation_time: float
    token_count: int
    prompt_used: str
    model_load_time: float
    prompt_creation_time: float
    total_response_time: float


@dataclass
class ResponseGenerator:
    llm: LocalLLM

    @classmethod
    def create(cls, llm: LocalLLM | None = None) -> "ResponseGenerator":
        configuration = load_configuration()
        model = llm or load_local_llm(model_candidates=configuration.llm_model_candidates)
        return cls(llm=model)

    def generate(
        self,
        question: str,
        retrieved_documents: Iterable[Any],
        retrieved_cases: Iterable[Any],
    ) -> dict[str, Any]:
        total_start = time.perf_counter()
        documents_list = _materialize_items(retrieved_documents)
        cases_list = _materialize_items(retrieved_cases)
        prompt_start = time.perf_counter()
        prompt = build_prompt(question, documents_list, cases_list)
        prompt_creation_time = time.perf_counter() - prompt_start
        if not documents_list and not cases_list:
            response = GenerationResponse(
                answer=INSUFFICIENT_EVIDENCE_RESPONSE,
                model_name=self.llm.model_name,
                generation_time=0.0,
                token_count=0,
                prompt_used=prompt,
                model_load_time=self.llm.load_seconds,
                prompt_creation_time=prompt_creation_time,
                total_response_time=time.perf_counter() - total_start,
            )
            return asdict(response)

        logger.info("Generation start: model=%s", self.llm.model_name)
        generation_settings = load_configuration().llm_generation_settings
        generation_text, token_count, generation_time = self.llm.generate(
            prompt=prompt,
            max_new_tokens=generation_settings.max_new_tokens,
            temperature=generation_settings.temperature,
            top_p=generation_settings.top_p,
            do_sample=generation_settings.do_sample,
        )
        logger.info("Generation finish: latency=%.4fs tokens=%s", generation_time, token_count)

        answer = _clean_output(generation_text)
        if not answer:
            answer = INSUFFICIENT_EVIDENCE_RESPONSE

        response = GenerationResponse(
            answer=answer,
            model_name=self.llm.model_name,
            generation_time=generation_time,
            token_count=token_count,
            prompt_used=prompt,
            model_load_time=self.llm.load_seconds,
            prompt_creation_time=prompt_creation_time,
            total_response_time=time.perf_counter() - total_start,
        )
        return asdict(response)


def generate_response(
    question: str,
    retrieved_documents: Iterable[Any],
    retrieved_cases: Iterable[Any],
    llm: LocalLLM | None = None,
) -> dict[str, Any]:
    generator = ResponseGenerator.create(llm=llm)
    return generator.generate(question, retrieved_documents, retrieved_cases)
    
def _materialize_items(items: Iterable[Any], max_items: int = 3) -> list[Any]:
    return list(items)[:max_items]


def _clean_output(text: str) -> str:
    return text.strip().replace("\u0000", "")