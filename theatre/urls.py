from django.urls import path

from . import views

app_name = "theatre"

urlpatterns = [
    path("", views.home, name="home"),
    path("productions/new/", views.new_production, name="new_production"),
    path("productions/<int:pk>/", views.production_detail, name="production_detail"),
    path("projects/", views.project_history, name="project_history"),
    path("rag-sources/", views.rag_sources, name="rag_sources"),
    path("projects/<int:pk>/rag-sources/", views.rag_sources, name="project_rag_sources"),
    path("research/", views.research_about, name="research_about"),
    path("research/rag/", views.research_rag, name="research_rag"),
    path("api/health/", views.health, name="health"),
]
