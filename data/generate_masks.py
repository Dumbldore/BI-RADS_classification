import os

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from models.segmentation import segmentation_model

FOLDERS = ["BUS"]
CLASSES = ["benign", "malignant"]
# CLASSES = [ "3", "4A", "4B", "4C", "5"]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform_input4test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((352, 352), antialias=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

model = segmentation_model.FCBFormer()
model.load_state_dict(
    torch.load("../checkpoints/FCBFormer_checkpoint.pt")["model_state_dict"]
)
model.to(DEVICE)
model.eval()

for folder in FOLDERS:
    for label in CLASSES:
        data_path = f"../images/full_image/{folder}/{label}/"
        save_path = f"../images/masks/{folder}/{label}/"
        os.makedirs(save_path, exist_ok=True)

        for image_name in os.listdir(data_path):
            image = cv2.imread(os.path.join(data_path, image_name))
            image_shape = image.shape

            image = transform_input4test(image).unsqueeze(0).to(DEVICE)
            output = torch.sigmoid(model(image))
            predicted_map = F.upsample(
                output,
                size=(image_shape[0], image_shape[1]),
                mode="bilinear",
                align_corners=False,
            )
            predicted_map = predicted_map.cpu().detach().numpy()
            predicted_map = np.squeeze(predicted_map)
            predicted_map = (predicted_map < 0.95) * predicted_map
            imageio.imwrite(save_path + image_name, predicted_map)
