"""Stage-lighting example retrieval."""
from theatre.services.data.schemas import ViewType

from .base import BaseRetriever


class LightingRetriever(BaseRetriever):
    view_type = ViewType.LIGHTING
    default_limit = 3
