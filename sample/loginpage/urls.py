# loginpage/urls.py

from django.urls import path
from loginpage import views  # Import the views from the loginpage app

urlpatterns = [
    path('', views.login, name='login'),  # Maps the base URL of homepage to the home view
]
