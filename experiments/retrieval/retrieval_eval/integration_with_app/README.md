# Retrieval Eval Integration With App

This subtree is intentionally allowed to call the canonical backend app in `apps/api`.

Use this folder for scripts that need:

- `from src...` imports
- Neo4j access via the backend service layer
- AQT or retrieval execution against the current application stack

Keep standalone post-processing in `retrieval_eval/run/` instead.
