# Learning Notebooks

This folder contains guided notebooks to test MCP tools and the LangGraph agent.

## Notebook Index

- `learning/01_mcp_tool_testing.ipynb`
  - Tests each MCP tool independently (`load_csv`, `calc_metrics`, `fetch_rules`, `query_graph`)
  - Measures per-tool latency
  - Helps isolate tool-level bugs before graph-level debugging

- `learning/02_langgraph_agent_testing.ipynb`
  - Runs end-to-end LangGraph analysis
  - Validates fallback behavior (Neo4j down, Ollama timeout)
  - Surfaces errors/warnings and explanation completeness checks

## Setup

1. Activate venv:

```powershell
.\.venv\Scripts\Activate.ps1
```

2. Install notebook runtime if needed:

```powershell
python -m pip install jupyter ipykernel
```

3. Start Jupyter:

```powershell
python -m jupyter lab
```

## Pinpointing Issues Quickly

1. Start with `01_mcp_tool_testing.ipynb`:
   - If a tool fails there, fix tool code first.
2. Then run `02_langgraph_agent_testing.ipynb`:
   - Inspect `payload["metadata"]["errors"]` and `payload["metadata"]["warnings"]`.
3. Confirm output integrity:
   - All SKUs have `plain_english_explanation`
   - `context_source` and `graph_source` make sense for your environment.

## Optional CLI Cross-checks

```powershell
.\.venv\Scripts\python -m pytest tests\ -q
$env:NEO4J_ENABLED='false'; .\.venv\Scripts\python main.py --data data/inventory_mock.csv --format table
```
