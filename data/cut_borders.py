import os

import cv2
import imageio
import numpy as np

FOLDERS = ["BUS"]
CLASSES = ["benign", "malignant"]
# CLASSES = ["2", "3", "4A", "4B", "4C", "5"]
TRESH = 30

for folder in FOLDERS:
    for label in CLASSES:
        save_path = f"../images/cut_borders/{folder}/{label}/"
        os.makedirs(save_path, exist_ok=True)
        full_image_path = f"../images/full_image/{folder}/{label}/"
        mask_path = f"../images/masks/{folder}/{label}/"

        for img_name, mask_name in zip(
            os.listdir(full_image_path), os.listdir(mask_path)
        ):
            img = cv2.imread(os.path.join(full_image_path, img_name))
            mask = cv2.imread(os.path.join(mask_path, mask_name), cv2.IMREAD_GRAYSCALE)
            _, mask = cv2.threshold(
                mask, thresh=180, maxval=255, type=cv2.THRESH_BINARY
            )
            if mask.max() == 0:
                imageio.imwrite(os.path.join(save_path, img_name), img)
                continue
            array = mask
            H, W = array.shape
            left_edges = np.where(array.any(axis=1), array.argmax(axis=1), W + 1)
            flip_lr = cv2.flip(array, 1)  # 1 horz vert 0
            right_edges = W - np.where(
                flip_lr.any(axis=1), flip_lr.argmax(axis=1), W + 1
            )
            top_edges = np.where(array.any(axis=0), array.argmax(axis=0), H + 1)
            flip_ud = cv2.flip(array, 0)  # 1 horz vert 0
            bottom_edges = H - np.where(
                flip_ud.any(axis=0), flip_ud.argmax(axis=0), H + 1
            )
            leftmost = left_edges.min()
            rightmost = right_edges.max()
            topmost = top_edges.min()
            bottommost = bottom_edges.max()
            leftmost = leftmost - TRESH
            if leftmost < 0:
                leftmost = 0
            rightmost = rightmost + TRESH
            if rightmost > W:
                rightmost = W
            topmost = topmost - TRESH
            if topmost < 0:
                topmost = 0
            bottommost = bottommost + TRESH
            if bottommost > H:
                bottommost = H
            imageio.imwrite(
                os.path.join(save_path, img_name),
                img[topmost:bottommost, leftmost:rightmost],
            )
