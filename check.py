import cv2
import matplotlib.pyplot as plt

# Charger image et masque
image = cv2.imread("data/images/fake_001.png")
mask = cv2.imread("data/masks/fake_001.png", cv2.IMREAD_GRAYSCALE)

# Convertir BGR -> RGB pour affichage
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Affichage
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.title("Image")
plt.imshow(image)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Masque binaire")
plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.show()
