from rest_framework import serializers


from ml_api.models import AnalysisImage


class AnalysisImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = AnalysisImage
        fields = '__all__'
