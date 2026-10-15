from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os
import time
import uuid
import logging

import api as api_module
from database import init_db

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
logger = logging.getLogger("supermew.api")


def create_app() -> FastAPI:
    app = FastAPI(
        title="SuperMew Agent API",
        description="Streaming RAG Agent with hybrid retrieval, document ingestion and RBAC.",
        version=os.getenv("APP_VERSION", "0.1.0"),
    )

    @app.on_event("startup")
    async def _startup_init_db():
        init_db()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in os.getenv(
            "CORS_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000"
        ).split(",") if origin.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health/live", tags=["health"])
    async def liveness() -> dict:
        return {"status": "ok", "service": "supermew", "version": app.version}

    @app.get("/health/ready", tags=["health"])
    async def readiness() -> dict:
        checks = {"database": "ok"}
        try:
            from database import engine
            from sqlalchemy import text
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except Exception as exc:
            logger.warning("readiness check failed: %s", exc)
            checks["database"] = "unavailable"
        status = "ok" if all(value == "ok" for value in checks.values()) else "degraded"
        return {"status": status, "checks": checks, "version": app.version}

    # No-cache middleware for development
    @app.middleware("http")
    async def _no_cache(request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - started) * 1000:.2f}"
        path = request.url.path or ""
        if path == "/" or path.endswith((".html", ".js", ".css")):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    app.include_router(api_module.router)

    # serve frontend static files at root
    if FRONTEND_DIR.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 8000)))
