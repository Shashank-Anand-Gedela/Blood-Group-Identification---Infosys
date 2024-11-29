# profilepage/urls.py

from django.urls import path
from profilepage import views  # Import the views from the loginpage app

urlpatterns = [
    path('', views.profile, name='profile'),  # Maps the base URL of homepage to the home view
]
