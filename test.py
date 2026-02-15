import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import SimpleUNet   # ✅ on importe ton modèle

# ===== 1️⃣ Charger le modèle =====
model = SimpleUNet()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

# ===== 2️⃣ Charger une image test =====
image_path = "data/images/GF2_PMS1__L1A0000564539-MSS1_0_512_size512.jpg"  # ⚠️ Mets ici une vraie image
image = cv2.imread(image_path)

if image is None:
    raise ValueError(f"Image introuvable : {image_path}")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ⚠️ Resize obligatoire (même taille que l'entraînement)
image_rgb = cv2.resize(image_rgb, (256, 256))

# Normalisation
image_tensor = torch.tensor(image_rgb / 255.0, dtype=torch.float32)
image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)

# ===== 3️⃣ Prédiction =====
with torch.no_grad():
    output = model(image_tensor)
    output = torch.sigmoid(output)
    prediction = output.squeeze().numpy()
print("valeur min :", prediction.min())    
print("valeur max :", prediction.max())  
# Binarisation
prediction = (prediction > 0.5).astype(np.uint8)

# ===== 4️⃣ Affichage =====
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Image originale")
plt.imshow(image_rgb)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Masque prédit")
plt.imshow(prediction, cmap="gray")
plt.axis("off")

plt.show()

