from django.apps import AppConfig

from ml_sources import executors

class MlSimpleApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ml_simple_api'

    device = 'cpu'  # or 'cuda'

    executor_segmentation = executors.SegmentationExecutor(device=device)
    executor_classification = executors.ClassificationExecutor(device=device)
    executor_extract = executors.ExtractExecutor(device=device)
