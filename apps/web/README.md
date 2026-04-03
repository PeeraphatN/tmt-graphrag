# Web App

This is the canonical home for the Next.js frontend of the GraphRAG PoC.

## Runtime Contents

- `app/`: App Router UI
- `public/`: static assets
- `package.json`: frontend dependencies consolidated into the web app itself
- `.env.local.example`: browser-visible API base URL for local development

## Run Locally

```powershell
cd apps/web
Copy-Item .env.local.example .env.local
npm install
npm run dev
```

The UI starts at `http://localhost:3000` and calls the API using `NEXT_PUBLIC_API_BASE_URL`.
