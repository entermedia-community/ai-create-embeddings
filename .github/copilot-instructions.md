## Purpose

This file guides automated coding agents (Copilot-style) so they can be productive immediately in this repository.

Note: a quick scan of the workspace during generation showed no obvious source files (no `src/`, `pyproject.toml`, `package.json`, or `README.md`). If you (the human) have the project in a different path or haven't pushed code yet, tell the agent where the entrypoint lives.

## Quick agent checklist (run in order)

1. Repo scan: look for these files/dirs (stop at first hits):

```
README.md
pyproject.toml, requirements.txt, setup.py
package.json
src/, app/, services/, notebooks/, scripts/, tests/
Dockerfile, Makefile
.env, .env.example
```

2. If nothing found, ask the user a single clarifying question: "Where is the main code or intended language (Python/Node/etc.)?"

3. If Python is present, run (locally or ask the user to run):

```
python -m pip install -r requirements.txt
pytest -q
```

If Node is present, run:

```
npm ci
npm test
```

4. Find the embedding pipeline by searching for keywords: `embed`, `embedding`, `vector`, `openai`, `huggingface`, `faiss`, `pinecone`, `weaviate`, `chroma`. Typical file names: `embedding*.py`, `create_embeddings.py`, `embedder.py`, `services/embedding/*`, `integrations/*`.

## Architecture discovery hints

- Expect a 3-stage flow where present: ingest -> preprocess -> embed -> store. Look for files or folders that map to those stages, e.g. `ingest/`, `preprocess/`, `embed/`, `store/` or `vectorstore/`.
- Look for provider adapters under `integrations/` or `providers/` (e.g. `openai_adapter.py`, `hf_provider.js`) — these are the safest places to change provider-specific code.
- If a `Dockerfile` or `Makefile` exists, prefer following those targets for reproducible builds.

## Project-specific conventions (inference rules)

- Environment: expect secrets in `.env` or in CI secrets. Never attempt to print or commit secrets. If you find `os.environ['...']` or `process.env[...]`, assume the variable is configured outside the repo.
- CLI scripts usually live in `scripts/` or top-level `bin/` and follow `if __name__ == "__main__"` for Python.
- Tests live under `tests/` or `spec/` and use `pytest` for Python or `jest`/`vitest` for JS/TS. Favor adding a small test next to new code.

## Integration & external dependencies

- Common external services: OpenAI, Hugging Face, Pinecone, Chroma, FAISS, Weaviate. Detect provider libraries in `requirements.txt` or `package.json` and treat provider-specific code as an integration boundary.
- If you need to add credentials for testing, use a local-only `.env.test` and document required keys in `README.md` rather than committing them.

## What to change and how to propose it

- Make surgical, small changes. Each change should include a test or a smoke-run command in the PR description.
- Use clear commit messages and PR descriptions that identify the modified stage (ingest|preprocess|embed|store) and the provider impacted.

## Safety, privacy, and secrets

- Never exfiltrate or log credentials. If you detect a secret in the tree, stop and ask the human which files to redact.

## If the repo is empty or missing expected files

- Ask one focused question to the repo owner: "Do you want me to scaffold a minimal Python (or Node) project for creating embeddings, including a CLI, a provider adapter, and a test? If so, which provider (openai/huggingface/other)?"

## Example prompts to the human

- "Where is the project's entrypoint or main language?"
- "Do you have an existing vector store/service (Pinecone/Chroma/FAISS) you want to use?"
- "Can I run tests and install dependencies in this environment?"

## Files to reference when present

- `README.md` — summary & run instructions
- `pyproject.toml` / `requirements.txt` — Python deps
- `package.json` — Node deps & scripts
- `Dockerfile`, `Makefile` — build/run targets
- `scripts/`, `src/`, `integrations/`, `tests/` — code and tests

If anything in this guidance is unclear or you want the agent to take an initial action (scan, scaffold, or run tests), tell the agent which of those to do next.
