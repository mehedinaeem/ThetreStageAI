"""Local research-development settings for ThetreStageAI."""
from __future__ import annotations

import os
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def env_bool(name: str, default: bool = False) -> bool:
    """Read a conventional boolean value from the environment."""
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def env_list(name: str, default: str = "") -> list[str]:
    """Read a comma-separated environment value."""
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


def project_path(name: str, default: str) -> Path:
    """Resolve an environment-configured path relative to the project root."""
    raw_value = os.getenv(name, default).strip()
    if not raw_value:
        raise ImproperlyConfigured(f"{name} cannot be empty")
    configured = Path(raw_value).expanduser()
    path = configured if configured.is_absolute() else BASE_DIR / configured
    return path.resolve()


def managed_storage_path(name: str, default: str) -> Path:
    """Reject dangerously broad targets for application-managed writable storage."""
    path = project_path(name, default)
    if path == Path(path.anchor) or path == BASE_DIR:
        raise ImproperlyConfigured(f"{name} must reference a dedicated subdirectory")
    return path


SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "unsafe-local-development-key")
DEBUG = env_bool("DJANGO_DEBUG", True)
ALLOWED_HOSTS = env_list("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1")
if not DEBUG and SECRET_KEY == "unsafe-local-development-key":
    raise ImproperlyConfigured(
        "DJANGO_SECRET_KEY must be configured when DJANGO_DEBUG is false"
    )

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "theatre.apps.TheatreConfig",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "theatre.middleware.RequestSizeLimitMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "thetrestageai.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    }
]

WSGI_APPLICATION = "thetrestageai.wsgi.application"
ASGI_APPLICATION = "thetrestageai.asgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "bn"
TIME_ZONE = "Asia/Dhaka"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Bound form/multipart parsing before expensive retrieval or generation work begins.
DATA_UPLOAD_MAX_MEMORY_SIZE = int(os.getenv("DJANGO_MAX_REQUEST_BYTES", "1048576"))
DATA_UPLOAD_MAX_NUMBER_FIELDS = 100

# Safe defaults for both local research and an eventual HTTPS deployment.
X_FRAME_OPTIONS = "DENY"
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_REFERRER_POLICY = "same-origin"
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "Lax"
CSRF_COOKIE_SAMESITE = "Lax"
SECURE_SSL_REDIRECT = env_bool("DJANGO_SECURE_SSL_REDIRECT", False)
SESSION_COOKIE_SECURE = env_bool("DJANGO_SECURE_COOKIES", False)
CSRF_COOKIE_SECURE = env_bool("DJANGO_SECURE_COOKIES", False)
SECURE_HSTS_SECONDS = int(os.getenv("DJANGO_SECURE_HSTS_SECONDS", "0"))
SECURE_HSTS_INCLUDE_SUBDOMAINS = env_bool("DJANGO_SECURE_HSTS_INCLUDE_SUBDOMAINS", False)
SECURE_HSTS_PRELOAD = env_bool("DJANGO_SECURE_HSTS_PRELOAD", False)

DATA_ROOT = managed_storage_path("DATA_ROOT", "data")
QDRANT_PATH = managed_storage_path("QDRANT_PATH", "storage/qdrant")
THEATRE_DATASET_PATH = project_path(
    "THEATRE_DATASET_PATH",
    "Dataset/bangla_natok_500",
)
if QDRANT_PATH == THEATRE_DATASET_PATH or QDRANT_PATH.is_relative_to(THEATRE_DATASET_PATH):
    raise ImproperlyConfigured("QDRANT_PATH cannot be inside the source dataset directory")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
QDRANT_UPSERT_BATCH_SIZE = int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", "64"))
RAG_CONTEXT_MAX_CHARS = int(os.getenv("RAG_CONTEXT_MAX_CHARS", "24000"))
THETRESTAGEAI_LLM_PROVIDER = os.getenv(
    "THETRESTAGEAI_LLM_PROVIDER", "gemini"
).strip().lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
GEMINI_TIMEOUT_SECONDS = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "180"))
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "8192"))
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
THETRESTAGEAI_OLLAMA_URL = os.getenv(
    "THETRESTAGEAI_OLLAMA_URL", "http://localhost:11434"
).rstrip("/")
THETRESTAGEAI_LLM_MODEL = os.getenv("THETRESTAGEAI_LLM_MODEL", "qwen3:4b")
THETRESTAGEAI_LLM_TIMEOUT_SECONDS = int(
    os.getenv("THETRESTAGEAI_LLM_TIMEOUT_SECONDS", "900")
)
THETRESTAGEAI_LLM_NUM_PREDICT = int(
    os.getenv("THETRESTAGEAI_LLM_NUM_PREDICT", "3072")
)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "{asctime} {levelname} {name}: {message}",
            "style": "{",
        }
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "standard"},
    },
    "root": {
        "handlers": ["console"],
        "level": os.getenv("DJANGO_LOG_LEVEL", "INFO"),
    },
    "loggers": {
        "theatre.research.experiments": {
            "handlers": ["console"],
            "level": os.getenv("THETRESTAGEAI_EXPERIMENT_LOG_LEVEL", "INFO"),
            "propagate": False,
        }
    },
}
