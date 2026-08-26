"""ASGI entry point for ThetreStageAI."""
import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "thetrestageai.settings")
application = get_asgi_application()
