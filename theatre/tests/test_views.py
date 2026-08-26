from django.test import TestCase
from django.urls import reverse


class BasicViewTests(TestCase):
    def test_home_page_loads(self) -> None:
        response = self.client.get(reverse("theatre:home"))
        self.assertEqual(response.status_code, 200)

    def test_health_endpoint(self) -> None:
        response = self.client.get(reverse("theatre:health"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
