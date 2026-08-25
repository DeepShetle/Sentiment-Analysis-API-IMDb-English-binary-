import random
from locust import HttpUser, task, between
from sample_reviews import SAMPLE_REVIEWS


class SentimentAPIUser(HttpUser):
    wait_time = between(0.1, 0.5)  # mô phỏng khoảng nghỉ nhỏ giữa các request, giống traffic thật

    @task(9)
    def predict(self):
        text = random.choice(SAMPLE_REVIEWS)
        self.client.post("/predict", json={"text": text})

    @task(1)
    def health_check(self):
        self.client.get("/health")