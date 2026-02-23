from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import config, datasets, runs, inference, auth


def create_app() -> FastAPI:
    app = FastAPI(title="LLM Finetuning Workbench API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
    app.include_router(config.router, prefix="/api/config", tags=["config"])
    app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
    app.include_router(runs.router, prefix="/api/runs", tags=["runs"])
    app.include_router(inference.router, prefix="/api/infer", tags=["inference"])
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
