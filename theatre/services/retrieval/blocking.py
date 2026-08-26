"""Actor-blocking example retrieval."""
from theatre.services.data.schemas import ViewType

from .base import BaseRetriever


class BlockingRetriever(BaseRetriever):
    view_type = ViewType.BLOCKING
    default_limit = 3
