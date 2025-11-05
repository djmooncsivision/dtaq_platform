from fastapi import APIRouter, Depends

from ...config import Settings, get_settings


router = APIRouter()


@router.get("/heartbeat")
async def heartbeat(settings: Settings = Depends(get_settings)) -> dict[str, str]:
    """Basic liveness check for the data preparation integration."""
    return {
        "service": "data_preparation",
        "proxy_target": settings.data_preparation_url,
        "status": "ready",
    }
