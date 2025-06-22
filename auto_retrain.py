import os
import io
import zipfile
import urllib.request
import tarfile
import shutil
import scipy.io
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import requests

# ---------------------------
# SETUP
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_OUTPUT = "best_flower_model.pt"
DOWNLOAD_URL = "https://flower-upload-api.onrender.com/download-data"
BASE_DIR = "flowers"
TRAIN_DIR = os.path.join(BASE_DIR, "train")

# ---------------------------
# TRANSFORMS
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------------------
# 1. Try to download uploaded data
# ---------------------------
os.makedirs("training_data", exist_ok=True)

print("📦 Checking for uploaded training data...")
use_uploaded_data = False
try:
    response = requests.get(DOWNLOAD_URL)
    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall("training_data")
        if os.listdir("training_data"):
            print("✅ Extracted uploaded training data.")
            use_uploaded_data = True
except Exception as e:
    print("❌ Failed to fetch uploaded data:", e)

# ---------------------------
# 2. Fallback: Use 102flowers.tgz
# ---------------------------
if not use_uploaded_data:
    print("⬇️  No uploaded data found. Downloading 102flowers.tgz...")

    FLOWER_URL = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    LABELS_URL = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
    SETID_URL = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"

    urllib.request.urlretrieve(FLOWER_URL, "102flowers.tgz")
    urllib.request.urlretrieve(LABELS_URL, "imagelabels.mat")
    urllib.request.urlretrieve(SETID_URL, "setid.mat")

    with tarfile.open("102flowers.tgz") as tar:
        tar.extractall()

    labels = scipy.io.loadmat("imagelabels.mat")["labels"][0]
    setid = scipy.io.loadmat("setid.mat")
    train_ids = setid["trnid"][0]
    val_ids = setid["valid"][0]

    def prepare_split(image_ids, target_dir):
        os.makedirs(target_dir, exist_ok=True)
        for i in image_ids:
            label = f"{labels[i-1]:03d}"
            label_dir = os.path.join(target_dir, label)
            os.makedirs(label_dir, exist_ok=True)
            src = os.path.join("jpg", f"image_{i:05d}.jpg")
            dst = os.path.join(label_dir, f"image_{i:05d}.jpg")
            shutil.copy(src, dst)

    prepare_split(train_ids, os.path.join(BASE_DIR, "train"))
    prepare_split(val_ids, os.path.join(BASE_DIR, "val"))

    print("✅ 102flowers dataset prepared.")

    DATA_DIR = os.path.join(BASE_DIR, "train")
else:
    DATA_DIR = "training_data"

# ---------------------------
# 3. Load dataset
# ---------------------------
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
num_classes = len(dataset.classes)
print(f"📊 Classes found: {num_classes}")

# ---------------------------
# 4. Build & train model
# ---------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(5):
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"📚 Epoch {epoch+1}: Loss={total_loss:.4f}")

# ---------------------------
# 5. Save TorchScript model
# ---------------------------
model.eval()
scripted_model = torch.jit.script(model)
scripted_model.save(MODEL_OUTPUT)
print(f"✅ Saved model as {MODEL_OUTPUT}")
