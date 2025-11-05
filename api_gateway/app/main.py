from fastapi import FastAPI

from .config import Settings, get_settings
from .api.routes import include_all_routers


def create_app(settings: Settings | None = None) -> FastAPI:
    """Initialize the FastAPI application with configured routes and metadata."""
    app_settings = settings or get_settings()
    app = FastAPI(
        title=app_settings.project_name,
        version=app_settings.version,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    include_all_routers(app)
    return app


app = create_app()
