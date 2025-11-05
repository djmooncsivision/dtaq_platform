from fastapi import FastAPI

from .routes import include_all_routers

__all__ = ["include_all_routers", "FastAPI"]
