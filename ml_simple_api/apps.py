import numpy as np
import pandas as pd

from django.apps import AppConfig

from ml_sources import executors

class MlSimpleApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ml_simple_api'

    device = 'cpu'  # or 'cuda'

    executor_segmentation = executors.SegmentationExecutor(device=device)
    executor_classification = executors.ClassificationExecutor(device=device)
    executor_extract = executors.ExtractExecutor(device=device)

    fv_dict = {}

    for i in range(9):
        features = np.load(f"media/class_array/{i}/all_f.npy")
        paths = np.load(f"media/class_array/{i}/all_p.npy")

        fv_dict[i] = {'features': features, 'paths': paths}

    db_csv = pd.read_csv('media/image_meta_data.csv')
