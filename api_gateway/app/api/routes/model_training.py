from fastapi import APIRouter, Depends

from ...config import Settings, get_settings


router = APIRouter()


@router.get("/heartbeat")
async def heartbeat(settings: Settings = Depends(get_settings)) -> dict[str, str]:
    """Basic liveness check for the model training integration."""
    return {
        "service": "model_training",
        "proxy_target": settings.model_training_url,
        "status": "ready",
    }
