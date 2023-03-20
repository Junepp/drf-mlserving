from typing import Optional

import torch
import keras

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'  # works on windows 10 to force it to use CPU


def load_segmentation(device: Optional[str] = 'cpu'):
    from segmentation_models_pytorch import Unet
    from iglovikov_helper_functions.dl.pytorch.utils import rename_layers

    device = torch.device(device)

    model = Unet(encoder_name="timm-efficientnet-b3", classes=1, encoder_weights=None)
    model = model.to(device)

    state_dict = torch.load('ml_sources/models/model_segmentation.pth', map_location=device)['state_dict']
    state_dict = rename_layers(state_dict, {"model.": ""})

    model.load_state_dict(state_dict)

    return model


def load_classification(device: Optional[str] = 'cpu'):
    model = keras.models.load_model('ml_sources/models/model_classification.h5', custom_objects=None, compile=True)

    return model


def load_fv_extactor(device: Optional[str] = 'cpu'):
    model = keras.models.load_model('ml_sources/models/model_fv_extractor.h5', custom_objects=None, compile=True)

    return model
