# Local runtime deployment

This directory provisions the GreenSpec RAG runtime floor only. It does not
implement or expose the frozen `/v1/retrieve` API before that service exists.

## One-time host setup

1. In Docker Desktop, open **Settings → Resources** and set Memory to at least
   10 GB (12 GB is preferred), then select **Apply & restart**.
2. Copy `.env.example` to `.env`. The default `RAG_RUNTIME_ROOT` stores Qdrant
   data and model weights at `D:\greenspec-rag-runtime`.
3. Keep the Docker engine in Linux containers mode.

## Provision and verify

```powershell
docker compose build
docker compose --profile diagnostics run --rm gpu-diagnostics
docker compose up -d qdrant
docker compose ps
Invoke-RestMethod http://127.0.0.1:6333/healthz
docker compose --profile setup run --rm model-download
```

`model-download` records exact resolved Hugging Face revisions in
`D:\greenspec-rag-runtime\models\model_manifest.json`. Commit or copy that
manifest into an index manifest during every index build.

## Security and persistence

Qdrant is bound only to `127.0.0.1`. Do not expose port 6333 or 6334 on a LAN
without authentication and network controls. Model weights remain outside the
repository; Qdrant uses the Docker-managed `greenspec-rag-qdrant-data` named
volume. Do not replace it with a Windows host bind mount: that configuration
previously corrupted persisted vector values.
