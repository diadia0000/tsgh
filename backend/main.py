"""FastAPI app entrypoint. Binds 127.0.0.1 only (docs/UI/01-architecture.md,
docs/UI/08-pitfalls-open-decisions.md #8) -- images never leave the machine."""
from fastapi import FastAPI

from backend.api import alignment, hybrid, jobs, tiles

app = FastAPI(title="tsgh backend")
app.include_router(alignment.router)
app.include_router(alignment.tus_router)
app.include_router(hybrid.router)
app.include_router(jobs.router)
app.include_router(tiles.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000)
