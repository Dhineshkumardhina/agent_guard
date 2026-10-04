"""FastAPI endpoint dependencies for session management and query parameters."""

from sqlalchemy.orm import Session
from fastapi import Depends

from backend.app.database.session import get_db
from backend.app.utils.pagination import PaginationParams

__all__ = [
    "get_db",
    "PaginationParams",
]
