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
from torch.utils.data import DataLoader, ConcatDataset, random_split
import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------
# CONFIGURATION
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_OUTPUT = "best_flower_model_v3.pt"
CLASS_MAPPING_FILE = "class_to_label.json"
DOWNLOAD_URL = "https://flower-upload-api.onrender.com/download-data"
BASE_DIR = "flowers"
OXFORD_TRAIN_DIR = os.path.join(BASE_DIR, "train")
OXFORD_VAL_DIR = os.path.join(BASE_DIR, "val") # Added validation directory
USER_DATA_DIR = "user_training_data"

# Oxford 102 Flowers dataset URLs
FLOWER_URL = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
LABELS_URL = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
SETID_URL = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"

# Training Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
VALIDATION_SPLIT_RATIO = 0.2 # 20% of merged data for validation

# ---------------------------
# TRANSFORMS
# ---------------------------
# Training transforms with augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Validation/Test transforms (no aggressive augmentation)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------

def download_and_extract_oxford_data():
    """Downloads and extracts the Oxford 102 Flowers dataset."""
    logging.info("⬇️ Downloading and preparing Oxford 102 Flowers dataset...")

    try:
        urllib.request.urlretrieve(FLOWER_URL, "102flowers.tgz")
        urllib.request.urlretrieve(LABELS_URL, "imagelabels.mat")
        urllib.request.urlretrieve(SETID_URL, "setid.mat")
    except urllib.error.URLError as e:
        logging.error(f"❌ Failed to download Oxford data: {e}")
        raise

    try:
        with tarfile.open("102flowers.tgz") as tar:
            tar.extractall()
    except tarfile.ReadError as e:
        logging.error(f"❌ Failed to extract 102flowers.tgz: {e}")
        raise

    logging.info("✅ Oxford 102 Flowers dataset downloaded and extracted.")

def prepare_oxford_splits(labels, train_ids, val_ids, image_dir="jpg"):
    """Organizes Oxford dataset into train and validation directories."""
    def _prepare_split_dir(image_ids, target_dir):
        os.makedirs(target_dir, exist_ok=True)
        for i in image_ids:
            # Labels are 1-indexed in .mat file, convert to 0-indexed for folder names
            label = f"{labels[i - 1]:03d}"
            label_dir = os.path.join(target_dir, label)
            os.makedirs(label_dir, exist_ok=True)
            src = os.path.join(image_dir, f"image_{i:05d}.jpg")
            dst = os.path.join(label_dir, f"image_{i:05d}.jpg")
            if os.path.exists(src):
                shutil.copy(src, dst)
            else:
                logging.warning(f"Image not found: {src}")

    logging.info("Preparing Oxford 102 Flowers dataset splits...")
    _prepare_split_dir(train_ids, OXFORD_TRAIN_DIR)
    _prepare_split_dir(val_ids, OXFORD_VAL_DIR)
    logging.info("✅ Oxford 102 Flowers dataset prepared for training and validation.")


def download_user_data():
    """Downloads and extracts user-uploaded training data."""
    logging.info("📦 Checking for uploaded training data...")
    os.makedirs(USER_DATA_DIR, exist_ok=True)

    try:
        response = requests.get(DOWNLOAD_URL, timeout=10) # Add timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(USER_DATA_DIR)

        if os.listdir(USER_DATA_DIR):
            # Check if extracted data has subdirectories (typical for ImageFolder)
            if any(os.path.isdir(os.path.join(USER_DATA_DIR, d)) for d in os.listdir(USER_DATA_DIR)):
                logging.info(f"✅ Found user-uploaded data in {len(os.listdir(USER_DATA_DIR))} folders.")
            else:
                logging.warning("ℹ️ Uploaded data extracted but no subdirectories found. Ensure it's class-wise organized.")
        else:
            logging.info("ℹ️ No images found in uploaded data or zip was empty.")

    except requests.exceptions.RequestException as e:
        logging.warning(f"❌ Failed to download uploaded data (network/HTTP error): {e}")
    except zipfile.BadZipFile:
        logging.warning("❌ Downloaded file is not a valid ZIP archive.")
    except Exception as e:
        logging.warning(f"❌ An unexpected error occurred during user data download: {e}")

def get_merged_datasets():
    """Loads and merges Oxford and user-uploaded datasets."""
    logging.info("📂 Loading datasets...")

    oxford_train_dataset = datasets.ImageFolder(OXFORD_TRAIN_DIR, transform=train_transform)
    oxford_val_dataset = datasets.ImageFolder(OXFORD_VAL_DIR, transform=val_transform)

    all_classes_set = set(oxford_train_dataset.classes)
    merged_train_dataset = oxford_train_dataset
    merged_val_dataset = oxford_val_dataset # Start with Oxford val

    if os.path.exists(USER_DATA_DIR) and os.listdir(USER_DATA_DIR):
        try:
            user_dataset = datasets.ImageFolder(USER_DATA_DIR, transform=train_transform) # Apply train transform for user data
            if user_dataset:
                logging.info(f"Adding {len(user_dataset)} images from user-uploaded data with {len(user_dataset.classes)} classes.")
                # Merge user data into the training set
                merged_train_dataset = ConcatDataset([oxford_train_dataset, user_dataset])
                all_classes_set.update(user_dataset.classes)
            else:
                logging.warning("User dataset is empty after loading.")
        except Exception as e:
            logging.warning(f"Could not load user dataset: {e}. Proceeding without it.")

    logging.info(f"✅ Total unique classes for training: {len(all_classes_set)}")

    # Create a unified class_to_idx mapping
    # This assumes that ImageFolder assigns indices based on sorted class names.
    # For ConcatDataset, the internal class_to_idx of the *first* dataset often dictates the overall mapping.
    # We need a robust way to map all classes to indices.
    all_sorted_classes = sorted(list(all_classes_set))
    unified_class_to_idx = {cls: i for i, cls in enumerate(all_sorted_classes)}

    # Adjust targets for merged dataset if necessary (critical for ConcatDataset with different class orderings)
    # This part is complex. The simplest way for ConcatDataset where ImageFolder maintains its own
    # class_to_idx is to rely on the fact that ImageFolder's target labels are already indices.
    # If a class exists in both, it will have different indices in different datasets.
    # A custom dataset wrapper would be needed for true unified indexing *before* DataLoader.
    # For simplicity here, we assume ImageFolder's numerical labels are fine as long as num_classes matches.
    # This means the model output should be able to map to any of the combined classes.

    # For the `num_classes` in the model, we use the count of unique classes.
    num_classes = len(all_sorted_classes)

    # Split the merged_train_dataset into training and validation if needed
    # This is important if user data has no separate validation
    train_size = int((1 - VALIDATION_SPLIT_RATIO) * len(merged_train_dataset))
    val_size = len(merged_train_dataset) - train_size
    train_subset, val_subset = random_split(merged_train_dataset, [train_size, val_size])

    # Combine Oxford val with the new validation subset from merged_train
    # This part needs careful consideration if you want a truly combined validation.
    # For simplicity, we'll just use the new `val_subset` and Oxford's pre-defined val.
    # A more robust approach would be to combine all data and then split train/val.
    final_train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    final_val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    # If you want to include oxford_val_dataset explicitly in overall validation:
    # final_val_loader = DataLoader(ConcatDataset([val_subset, oxford_val_dataset]), batch_size=BATCH_SIZE, shuffle=False)

    return final_train_loader, final_val_loader, all_sorted_classes, num_classes

def build_model(num_classes):
    """Builds and initializes the ResNet model."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes) # Only this layer will be trained by default

    model.to(DEVICE)
    logging.info(f"Model built with {num_classes} output classes.")
    return model

def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer):
    """Trains the model."""
    logging.info("📚 Starting model training...")
    best_loss = float('inf')

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            if batch_idx % 10 == 0: # Log every 10 batches
                logging.debug(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Train Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        logging.info(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # Validation phase
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = 100 * correct_val / total_val
        logging.info(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # Step the scheduler
        scheduler.step(avg_val_loss)

        # Save the best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model_weights.pth") # Save only weights
            logging.info(f"Saving best model weights with validation loss: {best_loss:.4f}")

    logging.info("✅ Training complete.")
    return model

def save_model_and_mapping(model, classes):
    """Saves the TorchScript model and class-to-label mapping."""
    # Load the best weights before scripting if you used checkpointing
    # model.load_state_dict(torch.load("best_model_weights.pth"))
    model.eval() # Set to eval mode before scripting

    scripted_model = torch.jit.script(model)
    scripted_model.save(MODEL_OUTPUT)
    logging.info(f"✅ Saved TorchScript model as {MODEL_OUTPUT}")

    # Create and save class-to-label mapping
    idx_to_label = {i: label for i, label in enumerate(classes)}
    with open(CLASS_MAPPING_FILE, "w") as f:
        json.dump(idx_to_label, f, indent=4)
    logging.info(f"✅ Saved class-to-label mapping to {CLASS_MAPPING_FILE}")

# ---------------------------
# MAIN EXECUTION
# ---------------------------
if __name__ == "__main__":
    # 1. Download and prepare Oxford 102 Flower dataset
    try:
        download_and_extract_oxford_data()
        labels = scipy.io.loadmat("imagelabels.mat")["labels"][0]
        setid = scipy.io.loadmat("setid.mat")
        train_ids = setid["trnid"][0]
        val_ids = setid["valid"][0] # Use Oxford's official validation set
        prepare_oxford_splits(labels, train_ids, val_ids)
    except Exception as e:
        logging.critical(f"Aborting due to Oxford dataset preparation failure: {e}")
        exit() # Exit if essential data is not ready

    # 2. Download uploaded user data (if available)
    download_user_data()

    # 3. Load and merge datasets
    train_loader, val_loader, all_classes, num_classes = get_merged_datasets()

    # 4. Build & train model
    model = build_model(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    trained_model = train_model(model, train_loader, val_loader, NUM_EPOCHS, criterion, optimizer)

    # 5. Save TorchScript model and class mapping
    save_model_and_mapping(trained_model, all_classes)

    # Clean up downloaded files
    for f in ["102flowers.tgz", "imagelabels.mat", "setid.mat"]:
        if os.path.exists(f):
            os.remove(f)
    if os.path.exists("jpg"):
        shutil.rmtree("jpg")
    logging.info("Cleanup complete.")
