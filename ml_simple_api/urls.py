from django.contrib import admin
from django.urls import path

from ml_simple_api import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('image/', views.CallModel.as_view())
]
