from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # Default Django admin interface
    path('admin/', admin.site.urls),
    
    # Connects your API endpoints (upload/ and predict/)
    # This matches the API_BASE_URL = 'http://localhost:8000/api' in your frontend
    path('api/', include('api.urls')),
]

# This allows the frontend to access and play the uploaded video files
# during development (required for the video preview to work)
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)