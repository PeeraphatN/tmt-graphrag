# API App

This is the staged home for the FastAPI and GraphRAG backend runtime.

## Runtime Contents

- `src/`: backend application code copied from the current runtime surface
- `manufacturers.json`: runtime lookup data required by manufacturer filtering
- `requirements.txt`: backend runtime dependencies
- `Dockerfile`: container entrypoint for the API app

## Run Locally

```powershell
cd apps/api
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item ..\..\infra\.env.example .env
python src/api/main.py
```

The API starts at `http://localhost:8000`.

## Notes

- NER is disabled by default in `infra/.env.example` so the backend can run without experiment artifacts.
- If you want NER enabled later, place the exported model under `artifacts/ner/final_model` and set `INTENT_V2_USE_NER=true`.
