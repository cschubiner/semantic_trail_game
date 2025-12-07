# Notes for Agents / Maintainers

- Always redeploy the Worker after any change under `worker/`:
  ```bash
  cd worker && npx wrangler deploy
  ```
  (You can also use `npm run deploy`.)
  Treat “change under worker/” as “deploy immediately.”

- Keep `ALLOWED_ORIGINS` in `worker/wrangler.toml` aligned with the frontend host (e.g., GitHub Pages URL) to avoid CORS issues.

- Frontend (`web/game.js`) must point `API_BASE` / `API_URL` to the deployed Worker and have `DEMO_MODE = false` for production.

- LLM ranking cache lives in `.llm_ranking_cache/` (gitignored); embedding cache is `.embedding_cache/`.

- GitHub Pages deploys `web/` via `.github/workflows/pages.yml`. Set Pages source to “GitHub Actions” in repo settings.
