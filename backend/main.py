"""FastAPI app entrypoint. Binds 127.0.0.1 by default (plain local dev); set
TSGH_HOST=0.0.0.0 to accept connections from outside the process -- needed in
Docker/EC2, where "images never leave the machine" (docs/UI/01-architecture.md,
docs/UI/08-pitfalls-open-decisions.md #8) no longer applies now that the app is
moving to a server deployment. Those two docs still assert the old rule and
need updating separately."""
import os

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

    uvicorn.run("backend.main:app", host=os.environ.get("TSGH_HOST", "127.0.0.1"), port=8000)
