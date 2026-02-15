import numpy as np
import cv2
import os

os.makedirs("data/images", exist_ok=True)
os.makedirs("data/masks", exist_ok=True)

# Image aléatoire
image = np.random.rdandint(0, 255, (256, 256, 3), dtype=np.uint8)

# Masque binaire (cercle blanc sur fond noir)
mask = np.zeros((256, 256), dtype=np.uint8)
cv2.circle(mask, (128, 128), 60, 255, -1)

cv2.imwrite("data/images/fake_001.png", image)
cv2.imwrite("data/masks/fake_001.png", mask)
