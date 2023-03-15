import torch


def load_segmentation():
    from segmentation_models_pytorch import Unet
    from iglovikov_helper_functions.dl.pytorch.utils import rename_layers

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = Unet(encoder_name="timm-efficientnet-b3", classes=1, encoder_weights=None)
    model = model.to(device)

    state_dict = torch.load('ml_sources/models/model_segmentation.pth', map_location=device)['state_dict']
    state_dict = rename_layers(state_dict, {"model.": ""})

    model.load_state_dict(state_dict)

    return model
