"""Main FastAPI web server mounting scientific API endpoints and static research UI."""

from __future__ import annotations

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes import router

app = FastAPI(
    title="Urban Microclimate Modelling - Neural Surrogates & CFD API",
    description="Backend API powering the Microclimate Research Overview and Field Comparison Studio.",
    version="1.4.0",
)

# Enable CORS for local research development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

# Mount static research UI files
ui_dir = Path(__file__).resolve().parent.parent / "ui"
ui_dir.mkdir(parents=True, exist_ok=True)

# Serve index.html on root
@app.get("/", include_in_schema=False)
def serve_index():
    index_file = ui_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Microclimate Research UI is initializing..."}

# Mount static directory for CSS, JS, assets
app.mount("/static", StaticFiles(directory=str(ui_dir)), name="static")


def start_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Entry point to launch the uvicorn web server."""
    import uvicorn

    uvicorn.run("src.api.server:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    start_server()
