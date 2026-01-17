from django.urls import path
from .views import UploadView, PredictView

urlpatterns = [
    path('upload/', UploadView.as_view()),
    path('predict/', PredictView.as_view()),
]