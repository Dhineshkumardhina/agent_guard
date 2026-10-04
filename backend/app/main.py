"""AgentGuard Research Platform FastAPI Application Entrypoint."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.app.core.config import settings
from backend.app.core.logging import configure_logging, get_logger
from backend.app.core.exceptions import AgentGuardAPIException
from backend.app.database.session import init_db, SessionLocal
from backend.app.services.db_seeder import seed_research_database
from backend.app.api.router import api_router
from backend.app.api.routes.health import router as health_router

configure_logging(settings.LOG_LEVEL)
_logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager initializing database tables and seeding data."""
    _logger.info("AgentGuard backend starting — initializing database tables")
    init_db()

    # Seed research data if needed
    db = SessionLocal()
    try:
        seed_research_database(db)
    except Exception as e:
        _logger.warning("Database seeder encountered an issue: %s", e)
    finally:
        db.close()

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
    description=(
        "Research REST API for AgentGuard: Temporal Graph-Based Detection "
        "and Prediction of Cascading Failures in Multi-Agent AI Systems."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Standardized error handlers
@app.exception_handler(AgentGuardAPIException)
async def agentguard_api_exception_handler(request: Request, exc: AgentGuardAPIException):
    """Handle custom AgentGuard API exceptions with consistent error envelope."""
    payload = {
        "error": {
            "code": exc.code,
            "message": exc.message,
        }
    }
    if exc.details is not None:
        payload["error"]["details"] = exc.details
    return JSONResponse(status_code=exc.status_code, content=payload)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle standard Starlette/FastAPI HTTP exceptions with consistent error envelope."""
    code_map = {
        404: "NOT_FOUND",
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        405: "METHOD_NOT_ALLOWED",
    }
    code = code_map.get(exc.status_code, f"HTTP_{exc.status_code}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": code, "message": str(exc.detail)}},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic request validation errors."""
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "INVALID_PARAMETER",
                "message": "Request validation failed. Check parameter types and bounds.",
                "details": exc.errors(),
            }
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Handle unexpected server errors without leaking internal stack traces."""
    _logger.exception("Unhandled server exception: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An internal server error occurred.",
            }
        },
    )


# Unversioned system endpoints
app.include_router(health_router)


@app.get("/", tags=["System"])
def root():
    """Root metadata endpoint."""
    return {
        "project": settings.PROJECT_NAME,
        "full_title": "AgentGuard: Temporal Graph-Based Detection and Prediction of Cascading Failures in Multi-Agent AI Systems",
        "version": settings.VERSION,
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "api_v1_prefix": "/api/v1",
    }


# Mount Version 1 research API routes
app.include_router(api_router)
