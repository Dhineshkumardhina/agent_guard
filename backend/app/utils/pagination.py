"""Pagination utilities and dependency helpers."""

from typing import TypeVar, Generic, Sequence, List
from pydantic import BaseModel, Field
from fastapi import Query

T = TypeVar("T")


class PaginationParams:
    """Dependency class for standard API pagination parameters."""

    def __init__(
        self,
        limit: int = Query(50, ge=1, le=200, description="Maximum number of items to return (1-200)"),
        offset: int = Query(0, ge=0, description="Offset index for pagination"),
    ):
        self.limit = limit
        self.offset = offset


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic wrapper for paginated collections."""
    items: List[T]
    total: int = Field(..., description="Total number of items matching filters")
    limit: int = Field(..., description="Pagination limit applied")
    offset: int = Field(..., description="Pagination offset applied")
    has_more: bool = Field(..., description="Whether more items are available beyond this page")


def paginate_sequence(items: Sequence[T], limit: int, offset: int) -> PaginatedResponse[T]:
    """Helper to paginate in-memory sequence or list."""
    total = len(items)
    sliced = list(items[offset : offset + limit])
    has_more = (offset + limit) < total
    return PaginatedResponse(
        items=sliced,
        total=total,
        limit=limit,
        offset=offset,
        has_more=has_more,
    )
