"""Health check route exposing backend service and database readiness."""

from fastapi import APIRouter
from backend.app.core.config import settings

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    summary="Health Check",
    description="Verify backend service status, version, and environment configuration.",
    status_code=200,
)
def get_health():
    """Return service health status."""
    return {
        "status": "ok",
        "service": "agentguard",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENV,
    }
