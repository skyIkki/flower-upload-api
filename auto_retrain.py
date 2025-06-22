import os
import io
import zipfile
import requests
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# 1. Download and extract the latest training data
DOWNLOAD_URL = "https://flower-upload-api.onrender.com/download-data"
print("📦 Downloading training data...")
response = requests.get(DOWNLOAD_URL)
if response.status_code != 200:
    raise Exception(f"Failed to download training data: {response.text}")
with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    zip_ref.extractall("training_data")
print("✅ Extracted training data.")

# 2. Setup training
DATA_DIR = "training_data"
MODEL_OUTPUT = "best_flower_model.pt"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 3. Build model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 4. Train
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.train()
for epoch in range(5):
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 5. Save model
model.eval()
scripted_model = torch.jit.script(model)
scripted_model.save(MODEL_OUTPUT)
print(f"✅ Saved new model: {MODEL_OUTPUT}")
