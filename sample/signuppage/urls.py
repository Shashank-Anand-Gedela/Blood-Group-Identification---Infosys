# loginpage/urls.py

from django.urls import path
from signuppage import views  # Import the views from the homepage app

urlpatterns = [
    path('', views.signup, name='signup'),  # Maps the base URL of homepage to the home view
]
