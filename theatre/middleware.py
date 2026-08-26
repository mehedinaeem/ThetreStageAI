"""Small request-boundary middleware for the local research application."""
from __future__ import annotations

from collections.abc import Callable

from django.conf import settings
from django.http import HttpRequest, HttpResponse


class RequestSizeLimitMiddleware:
    """Return a controlled 413 before Django parses an obviously oversized body."""

    def __init__(self, get_response: Callable[[HttpRequest], HttpResponse]) -> None:
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        raw_length = request.META.get("CONTENT_LENGTH", "")
        try:
            content_length = int(raw_length) if raw_length else 0
        except (TypeError, ValueError):
            content_length = 0
        limit = settings.DATA_UPLOAD_MAX_MEMORY_SIZE
        if content_length > limit:
            return HttpResponse(
                "Request body is too large.",
                status=413,
                content_type="text/plain; charset=utf-8",
            )
        return self.get_response(request)
