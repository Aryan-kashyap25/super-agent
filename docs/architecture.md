# LocalMind AI Architecture

## System Architecture

![System Architecture](../graph_workflow.png)

## Workflow

1. The CLI accepts a question and sends it into the LangGraph workflow.
2. The triage node classifies the request into answerable, clarification, escalation, or out of scope.
3. Conditional routing directs the request to the appropriate next node.
4. Answerable requests trigger retrieval across the knowledge base and resolved cases.
5. The generation node uses a local Hugging Face model to draft an evidence-grounded response.
6. The verification node validates evidence support, source structure, schema compliance, and confidence.
7. If verification fails, the graph retries generation exactly once.
8. If the retry also fails, the graph returns a safe failure response.

## Module Responsibilities

- `app.py`: Interactive CLI, banner rendering, command handling, execution reporting, and JSON output.
- `graph/`: LangGraph orchestration, routing, and state management.
- `retrieval/`: Document loading, chunking, embedding, FAISS indexing, and semantic search.
- `llm/`: Local Hugging Face model loading, prompt building, and response generation.
- `verifier/`: Evidence validation, schema validation, confidence scoring, and safe failure handling.
- `utils/`: Shared logging, helper, timer, and schema utilities.

## Operational Notes

- The project is local-first and does not call external cloud APIs.
- Retrieval and generation are lazily loaded in the CLI to keep startup responsive.
- Verification is deterministic and remains outside the LLM.
