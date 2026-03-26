# Infra

This directory holds the staged runtime wiring for the reorganized application.

## Files

- `docker-compose.yml`: local service boot for Neo4j, Ollama, API, and web
- `.env.example`: environment template aligned with the new application layout

## Run With Docker Compose

```powershell
Copy-Item infra\.env.example infra\.env
cd infra
docker compose up --build
```

The compose setup now targets `apps/api` and `apps/web`, and keeps NER disabled by default.
