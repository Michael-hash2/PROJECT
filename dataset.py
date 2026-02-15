import os
import cv2
import torch
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        # ===== IMAGE =====
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)

        if image is None:
            raise ValueError(f"Image introuvable : {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(256, 256))
        image = image / 255.0
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        # ===== MASQUE =====
        base_name = os.path.splitext(img_name)[0]
        mask_name = base_name + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        mask = cv2.imread(mask_path, 0)

        if mask is None:
            raise ValueError(f"Masque introuvable : {mask_path}")
        mask = cv2.resize(mask,(256, 256))

        mask = mask / 255.0
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask
