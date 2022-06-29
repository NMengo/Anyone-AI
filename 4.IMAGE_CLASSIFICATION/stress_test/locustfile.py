from importlib.metadata import files
from locust import HttpUser, task, between

class UserBehavior(HttpUser):
    wait_time = between(1, 5)

    @task(30)
    def index_page(self):
        self.client.get("/")

    @task(10)
    def press_page(self):
        with open('dog.jpeg', 'rb') as image:
            self.client.post("/predict", files={'file':image})
