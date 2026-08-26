"""Scene-example retrieval."""
from theatre.services.data.schemas import ViewType

from .base import BaseRetriever


class SceneRetriever(BaseRetriever):
    view_type = ViewType.SCENE
    default_limit = 5
