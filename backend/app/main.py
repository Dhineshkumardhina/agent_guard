"""AgentGuard Research Platform FastAPI Application Entrypoint."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import settings
from backend.app.core.logging import configure_logging, get_logger
from backend.app.database.session import init_db

configure_logging(settings.LOG_LEVEL)
_logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager initializing database tables."""
    _logger.info("AgentGuard backend starting — initializing database tables")
    init_db()
    _logger.info(
        "Startup complete | project=%s version=%s env=%s",
        settings.PROJECT_NAME,
        settings.VERSION,
        settings.ENV,
    )
    yield
    _logger.info("AgentGuard backend shutting down")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    lifespan=lifespan,
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"])
def health_check():
    """Verify backend and database readiness."""
    return {
        "status": "ok",
        "service": "agentguard",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENV,
    }


@app.get("/", tags=["System"])
def root():
    """Root metadata endpoint."""
    return {
        "project": settings.PROJECT_NAME,
        "full_title": "AgentGuard: Temporal Graph-Based Detection and Prediction of Cascading Failures in Multi-Agent AI Systems",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
    }
