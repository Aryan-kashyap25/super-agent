from __future__ import annotations

import app


class FakeGraph:
    def invoke(self, state):
        return {
            "classification": "clarification",
            "answer": "I need a little more information before I can answer your question.",
            "sources": [],
            "confidence": 0.0,
            "requires_human": False,
            "reason": "Needs clarification.",
            "logs": ["START", "TRIAGE", "ROUTING", "CLARIFICATION", "END"],
            "execution_path": ["START", "TRIAGE", "ROUTING", "CLARIFICATION", "END"],
            "metadata": {},
            "generation_metadata": {},
            "verification_metadata": {},
            "retry_count": 0,
        }


class FakeConsole:
    def __init__(self):
        self.messages = []
        self._inputs = iter(["exit"])

    def print(self, *args, **kwargs):
        self.messages.append((args, kwargs))

    def input(self, prompt):
        return next(self._inputs)

    def clear(self):
        return None


def test_cli_startup(monkeypatch):
    monkeypatch.setattr(app, "build_support_graph", lambda dependencies: FakeGraph())
    monkeypatch.setattr(app, "Console", FakeConsole)
    assert app.main() == 0