from rest_framework import routers
from django.urls import include, path

from ml_api.views import AnalysisImageViewSet


router = routers.DefaultRouter()
router.register(r'image', AnalysisImageViewSet)

urlpatterns = [path('', include(router.urls))]
