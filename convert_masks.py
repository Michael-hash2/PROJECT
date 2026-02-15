import os
import cv2
import numpy as np

input_folder = "data/masks"
output_folder = "data/masks_binary"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):

    mask_path = os.path.join(input_folder, filename)
    mask = cv2.imread(mask_path)

    if mask is None:
        continue

    # Plage large de vert (plus robuste)
    lower_green = np.array([0, 150, 0])      # BGR
    upper_green = np.array([120, 255, 120])  # BGR

    binary_mask = cv2.inRange(mask, lower_green, upper_green)

    save_path = os.path.join(output_folder, filename)
    cv2.imwrite(save_path, binary_mask)

    print(filename, np.unique(binary_mask))

print("Conversion terminée.")

