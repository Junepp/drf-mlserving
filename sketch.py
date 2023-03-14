from segmentation_models_pytorch import Unet

model = Unet(encoder_name="timm-efficientnet-b3", classes=1, encoder_weights=None)

print(model)
