import cv2
import torch
import albumentations as albu
import numpy as np
from ml_sources.load_models import load_segmentation
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad

model = load_segmentation()

img_path = 'media/어반로지97.jpg'

img = load_rgb(img_path)
padded_img, pads = pad(img, factor=32, border=cv2.BORDER_CONSTANT)

transform = albu.Compose([albu.Normalize(p=1)], p=1)
x = transform(image=padded_img)["image"]
x = torch.unsqueeze(tensor_from_rgb_image(x), 0)

with torch.no_grad():
    prediction = model(x)[0][0]

mask = (prediction > 0).cpu().numpy().astype(np.uint8)
mask = unpad(mask, pads)
mask = mask.reshape(mask.shape[0], mask.shape[1], 1)

result = cv2.bitwise_and(img, img, mask=mask)
result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

cv2.imshow('t', result)
cv2.waitKey()

# for i in range(mask.shape[2]):
#     temp = skimage.io.imread(input_image)
#     for j in range(temp.shape[2]):
#         temp[:, :, j] = temp[:, :, j] * mask[:, :, i]
#     plt.figure(figsize=(8, 8))

# skimage.io.imsave('static/segmentation_img/' + str(timestr) + '.png', temp)
