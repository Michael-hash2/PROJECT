import torch
from torch.utils.data import DataLoader
from dataset import SegmentationDataset
from model import SimpleUNet
import torch.nn as nn
import torch.optim as optim

# chemins
image_dir = "data/images"
mask_dir = "data/masks_binary"

# dataset
dataset = SegmentationDataset(image_dir, mask_dir)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# modèle
model = SimpleUNet()

# loss
criterion = nn.BCELoss()

# optimiseur
optimizer = optim.Adam(model.parameters(), lr=0.001)

# entraînement
epochs = 50

for epoch in range(epochs):
    total_loss = 0

    for images, masks in loader:
        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader)}")

torch.save(model.state_dict(), "model.pth")
print("Modèle entraîné et sauvegardé !")
