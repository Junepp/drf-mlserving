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
        image_array = tensorflow.keras.utils.img_to_array(image)
        image_batch = np.array([image_array])

        result = self.model.predict(image_batch, verbose=0)
        result_class = np.argmax(result)

        index_to_str = ['바캉스','보헤미안','섹시','스포티','오피스룩','캐주얼','트레디셔널','페미닌','힙합'][result_class]

        return index_to_str


class ExtractExecutor:

    def __init__(self, device: Optional[str] = 'cpu'):
        self.device = device
        self.model = load_models.load_fv_extactor()

    def execute(self):

        return
