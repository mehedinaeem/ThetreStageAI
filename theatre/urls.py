from django.urls import path

from . import views

app_name = "theatre"

urlpatterns = [
    path("", views.home, name="home"),
    path("api/health/", views.health, name="health"),
]
