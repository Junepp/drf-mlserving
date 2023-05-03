import cv2
import torch
import numpy as np
import albumentations as albu
from PIL import Image
from typing import Optional

import tensorflow

from ml_sources import load_models
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from iglovikov_helper_functions.utils.image_utils import pad, unpad


class SegmentationExecutor:

    def __init__(self, device: Optional[str] = 'cpu'):
        self.device = device
        self.model = load_models.load_segmentation(device=device)

    def execute(self, image: np.ndarray) -> Image.Image:
        padded_img, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)
        transform = albu.Compose([albu.Normalize(p=1)], p=1)

        image_array = transform(image=padded_img)["image"]
        image_array = torch.unsqueeze(tensor_from_rgb_image(image_array), 0)
        image_array = image_array.to(self.device)

        with torch.no_grad():
            prediction = self.model(image_array)[0][0]

        mask = (prediction > 0).cpu().numpy().astype(np.uint8)
        mask = unpad(mask, pads)
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)

        result = cv2.bitwise_and(image, image, mask=mask)
        result = Image.fromarray(result)

        return result


class ClassificationExecutor:

    def __init__(self, device: Optional[str] = 'cpu'):
        self.device = device
        self.model = load_models.load_classification()

    def execute(self, image: Image.Image):
        image = image.resize(size=(128, 256))

        x = tensorflow.keras.utils.img_to_array(image)
        x = np.array([x])

        prediction = self.model.predict(x, verbose=0)
        prediction_argmax = np.argmax(prediction)

        # label = ['바캉스','보헤미안','섹시','스포티','오피스룩','캐주얼','트레디셔널','페미닌','힙합'][prediction]

        return prediction, prediction_argmax


class ExtractExecutor:

    def __init__(self, device: Optional[str] = 'cpu'):
        self.device = device
        self.model = load_models.load_fv_extactor()

    def execute(self, image: Image.Image):
        image = image.resize(size=(224, 224))
        # image = image.convert('RGB')

        x = tensorflow.keras.utils.img_to_array(image)
        x = np.expand_dims(x, axis=0)
        # x = np.array([x])
        # x = tensorflow.keras.applications.imagenet_utils(x)

        feature_vector = self.model.predict(x)[0]
        feature_vector = feature_vector / np.linalg.norm(feature_vector)

        return feature_vector
