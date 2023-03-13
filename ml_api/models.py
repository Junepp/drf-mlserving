from django.db import models


class AnalysisImage(models.Model):
    image = models.ImageField(blank=True, null=True)
