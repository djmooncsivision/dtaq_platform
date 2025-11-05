from fastapi import APIRouter, Depends

from ...config import Settings, get_settings


router = APIRouter()


@router.get("/heartbeat")
async def heartbeat(settings: Settings = Depends(get_settings)) -> dict[str, str]:
    """Basic liveness check for the model evaluation integration."""
    return {
        "service": "model_evaluation",
        "proxy_target": settings.model_evaluation_url,
        "status": "ready",
    }
