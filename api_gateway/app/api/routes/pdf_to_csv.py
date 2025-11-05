from fastapi import APIRouter, Depends

from ...config import Settings, get_settings


router = APIRouter()


@router.get("/heartbeat")
async def heartbeat(settings: Settings = Depends(get_settings)) -> dict[str, str]:
    """Basic liveness check for the PDF-to-CSV conversion microservice."""
    return {
        "service": "dtaq_func_pdf_to_csv",
        "proxy_target": settings.pdf_to_csv_url,
        "status": "ready",
    }
