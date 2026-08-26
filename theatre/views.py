"""Thin HTTP views; domain work belongs in theatre.services."""
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response


def home(request: HttpRequest) -> HttpResponse:
    return render(request, "theatre/home.html")


@api_view(["GET"])
def health(_: HttpRequest) -> Response:
    return Response({"status": "ok", "service": "ThetreStageAI"})
