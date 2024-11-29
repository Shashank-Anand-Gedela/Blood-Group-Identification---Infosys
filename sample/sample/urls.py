# sample/urls.py (Main project URLs)

from django.contrib import admin
from django.urls import path, include  # Use include to reference app URLs

urlpatterns = [
    path('admin/', admin.site.urls),             # Admin URL
    path('', include('homepage.urls')),          # Include homepage URLs for the base URL
    path('demopage/', include('demopage.urls')), # Include demopage URLs for the /demopage/ URL
    path('login/', include('loginpage.urls')), # Include loginpage URLs for the /loginpage/ URL
    path('signup/', include('signuppage.urls')), # Include signuppage URLs for the /signuppage/ URL
    path('profile/',include('profilepage.urls')),# Include profilepage URLs for the /profilepage/ URL

]

