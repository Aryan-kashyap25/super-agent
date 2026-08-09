# LocalMind AI

LocalMind AI is a local-first intelligent support agent built for an internship submission. It uses a Retrieval Engine, a local Hugging Face LLM, LangGraph orchestration, and a deterministic verification layer to produce evidence-grounded answers from the offline knowledge base.

## Project Overview

The project is organized as a phased Python application:

- Phase 1: Project foundation and runtime setup
- Phase 2: Retrieval Engine
- Phase 3: Local LLM Generation Engine
- Phase 4: LangGraph Orchestration
- Phase 5: Verification Layer
- Phase 6: Final integration, CLI, documentation, and submission polishing

## Features

- Local-first execution with no cloud API dependencies
- Knowledge-base retrieval from markdown files and resolved cases
- Local Hugging Face text generation
- LangGraph-based workflow orchestration
- Deterministic verification with a single retry
- Interactive CLI with execution trace, retrieved sources, human-readable output, and formatted JSON
- Rich logging and performance reporting
- Submission-ready docs, sample outputs, and architecture notes

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full system architecture, workflow, and module responsibilities.

### Workflow

![Graph Workflow](graph_workflow.png)

## Folder Structure

```text
LocalMind AI/
├── app.py
├── config.py
├── README.md
├── graph_workflow.png
├── .gitignore
├── .env.example
├── requirements.txt
g
├── data/
│   ├── knowledge_base/
│   ├── resolved_cases.json
│   └── sample_questions.json
├── docs/
│   └── architecture.md
├── graph/
│   ├── __init__.py
│   ├── state.py
│   ├── router.py
│   ├── edges.py
│   └── graph.py
├── retrieval/
│   ├── __init__.py
│   ├── loader.py
│   ├── chunker.py
│   ├── embedding.py
│   ├── vector_store.py
│   └── search.py
├── llm/
│   ├── __init__.py
│   ├── loader.py
│   ├── prompts.py
│   └── generator.py
├── verifier/
│   ├── __init__.py
│   ├── schema_validator.py
│   ├── evidence_checker.py
│   ├── confidence.py
│   └── verifier.py
├── utils/
│   ├── __init__.py
│   ├── logger.py
│   ├── helpers.py
│   ├── timer.py
│   └── schema.py
├── models/
├── outputs/
│   ├── logs/
│   │   └── sample_execution.log
│   ├── samples/
│   │   ├── answerable.json
│   │   ├── clarification.json
│   │   ├── multi_document.json
│   │   ├── out_of_scope.json
│   │   └── verification_failure.json
│   └── screenshots/
│       └── placeholder.md
└── tests/
    ├── test_cli.py
    ├── test_generator.py
    ├── test_graph.py
    ├── test_retrieval.py
    └── test_verification.py
```

## Installation

### Virtual Environment

Create and activate a virtual environment named `.venv`.

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### Requirements & Setup

Follow these step-by-step instructions to run the project locally:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python app.py
   ```

## Hardware & Models Used

**Hardware Used:**
- Intel i5, 16 GB RAM, NO GPU

**Models Used:**
- **LLM Generator:** `Qwen/Qwen2.5-0.5B-Instruct` (Primary) / `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (Fallback)
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`

The CLI loads models lazily to keep startup fast. If a model is not cached locally, the application will download it on the first run.

## How to Run

Start the interactive CLI:

```bash
python app.py
```

Available commands:

- `exit`
- `quit`
- `clear`

## Example Usage

```text
Question > What do the docs say about API credentials?
```

The application will display:

- Project banner
- Model and device information
- Execution trace
- Retrieved sources
- Final answer
- Structured JSON output
- Performance metrics

## Performance Metrics

The CLI reports these metrics after every query:

- Model load time
- Embedding load time
- Retrieval latency
- Generation latency
- Verification latency
- Total response time

## Design Trade-offs & Limitations

- **Hardware Constraint:** Used a smaller LLM (Qwen 0.5B) for CPU compatibility, which sometimes generates slightly broken JSON, handled via regex during output parsing.
- The project is intentionally scoped to offline support documentation.
- Retrieval quality depends on the quality of the knowledge base content.
- The current verification layer is deterministic and conservative by design.
- If local model files are unavailable, the CLI will surface a graceful failure instead of calling any external service.

## Future Improvements

- Add richer evaluation datasets
- Add submission utilities for automated artifact capture
- Expand the verification heuristics
- Improve source ranking and evidence tracing
- Add packaging and deployment hardening

## AI Disclosure

Used AI tools for LangGraph boilerplate and regex generation. The implementation is intended for educational and internship submission purposes.

## License

No license has been declared for this internship submission.
