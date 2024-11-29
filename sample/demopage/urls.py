# demopage/urls.py

from django.urls import path
from demopage import views  # Import the views from the demopage app

urlpatterns = [
    path('', views.home, name='home'),  # Maps the base URL of demopage to the index view
]
