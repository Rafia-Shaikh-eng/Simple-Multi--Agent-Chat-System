# Simple Multi-Agent Chat System

A minimal multi-agent system where a **Coordinator (Manager)** orchestrates **Research**, **Analysis**, and **Memory** agents to answer user questions with **traceable collaboration** and a **structured memory layer** (conversation, knowledge base, agent state). Implements **vector search** (TF-cosine) with no external dependencies.

> Built per the Technical Assessment brief. See the PDF for full requirements. (Implements: Coordinator, ResearchAgent, AnalysisAgent, MemoryAgent, vector store, planning, logging, test scenarios, outputs folder.)

## Architecture

- **Coordinator**
  - Classifies query complexity (`simple`, `memory`, `complex`)
  - Plans steps (reuse check → research → analysis, etc.)
  - Routes messages to agents, merges results, persists findings
  - Writes trace logs to `outputs/trace.log`
- **ResearchAgent**
  - Simulated retrieval over a preloaded `MOCK_KB` (see `app/knowledge_base.py`)
- **AnalysisAgent**
  - Performs simple comparisons/reasoning/summaries over research output
- **MemoryAgent**
  - Stores conversation transcripts
  - Knowledge Base with vector search and structured metadata (topic, source, confidence, timestamps)
  - Agent State memory to record plans/outcomes
  - Suggests reuse to avoid redundant work
- **Vector Store**
  - In-memory TF-cosine similarity (no FAISS/Chroma dependency)
- **Tracing**
  - Each step appended to a `trace` payload; also appended to `outputs/trace.log`

### Sequence Flow (typical complex query)
1. Coordinator receives user message → classifies as `complex`
2. `reuse_check`: search prior knowledge by vector similarity
3. `research`: retrieve related items
4. `analysis`: reason over research results
5. Coordinator synthesizes final answer, persists knowledge + agent state

## Getting Started

### Local (Python 3.10+)
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tests/run_scenarios.py
python main.py   # optional: interactive console
