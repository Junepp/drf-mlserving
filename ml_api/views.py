from rest_framework import viewsets

from ml_api.models import AnalysisImage
from ml_api.serializers import AnalysisImageSerializer



class AnalysisImageViewSet(viewsets.ModelViewSet):
    queryset = AnalysisImage.objects.all()
    serializer_class = AnalysisImageSerializer
