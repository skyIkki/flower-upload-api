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
from torch.utils.data import DataLoader, ConcatDataset
import requests

# ---------------------------
# SETUP
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_OUTPUT = "best_flower_model_v3.pt"
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
# 1. Download and prepare Oxford 102 Flower dataset
# ---------------------------
print("⬇️ Downloading and preparing Oxford 102 Flowers dataset...")

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
        label = f"{labels[i - 1]:03d}"
        label_dir = os.path.join(target_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        src = os.path.join("jpg", f"image_{i:05d}.jpg")
        dst = os.path.join(label_dir, f"image_{i:05d}.jpg")
        shutil.copy(src, dst)

prepare_split(train_ids, os.path.join(BASE_DIR, "train"))
print("✅ Oxford 102 Flowers dataset prepared.")

# ---------------------------
# 2. Download uploaded user data (if available)
# ---------------------------
print("📦 Checking for uploaded training data...")
user_data_dir = "user_training_data"
os.makedirs(user_data_dir, exist_ok=True)

try:
    response = requests.get(DOWNLOAD_URL)
    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(user_data_dir)
        if os.listdir(user_data_dir):
            print(f"✅ Found user-uploaded data: {len(os.listdir(user_data_dir))} folders.")
        else:
            print("ℹ️ No images in uploaded data.")
    else:
        print("❌ Failed to download uploaded data (non-200).")
except Exception as e:
    print("❌ Failed to download uploaded data:", e)

# ---------------------------
# 3. Load and merge datasets
# ---------------------------
print("📂 Loading datasets...")

oxford_dataset = datasets.ImageFolder(os.path.join(BASE_DIR, "train"), transform=transform)

# Merge with user-uploaded dataset (if exists)
if os.path.exists(user_data_dir) and os.listdir(user_data_dir):
    user_dataset = datasets.ImageFolder(user_data_dir, transform=transform)

    # Combine Oxford + user-uploaded
    merged_dataset = ConcatDataset([oxford_dataset, user_dataset])
    print(f"✅ Total combined classes: {len(set(oxford_dataset.classes + user_dataset.classes))}")
else:
    merged_dataset = oxford_dataset
    print(f"✅ Total Oxford classes only: {len(oxford_dataset.classes)}")

loader = DataLoader(merged_dataset, batch_size=32, shuffle=True)

# ---------------------------
# 4. Build & train model
# ---------------------------
num_classes = len(set(oxford_dataset.classes + (user_dataset.classes if 'user_dataset' in locals() else [])))

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
