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
import random # For setting random seeds

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seeds for reproducibility
def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    # np.random.seed(seed) # If you use numpy
set_all_seeds(42) # You can choose any integer seed

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
# OXFORD CODE TO COMMON NAME MAPPING
# This map translates the numeric folder names (e.g., "001") to human-readable names.
# This list is obtained from official Oxford 102 Flower dataset documentation or external resources.
# Ensure this list is complete and matches the order/mapping used by the dataset provider.
# Your provided Java snippet contains the correct mapping, replicated here.
# ---------------------------
OXFORD_CODE_TO_NAME_MAP = {
    "001": "pink primrose", "002": "globe thistle", "003": "blanket flower", "004": "trumpet creeper",
    "005": "blackberry lily", "006": "snapdragon", "007": "colt's foot", "008": "king protea",
    "009": "spear thistle", "010": "yellow iris", "011": "globe-flower", "012": "purple coneflower",
    "013": "peruvian lily", "014": "balloon flower", "015": "hard-leaved pocket orchid",
    "016": "giant white arum lily", "017": "fire lily", "018": "pincushion flower", "019": "fritillary",
    "020": "red ginger", "021": "grape hyacinth", "022": "corn poppy", "023": "prince of wales feathers",
    "024": "stemless gentian", "025": "artichoke", "026": "canterbury bells", "027": "sweet william",
    "028": "carnation", "029": "garden phlox", "030": "love in the mist", "031": "mexican aster",
    "032": "alpine sea holly", "033": "ruby-lipped cattleya", "034": "cape flower", "035": "great masterwort",
    "036": "siam tulip", "037": "sweet pea", "038": "lenten rose", "039": "barberton daisy",
    "040": "daffodil", "041": "sword lily", "042": "poinsettia", "043": "bolero deep blue",
    "044": "wallflower", "045": "marigold", "046": "buttercup", "047": "oxeye daisy",
    "048": "english marigold", "049": "common dandelion", "050": "petunia", "051": "wild pansy",
    "052": "primula", "053": "sunflower", "054": "pelargonium", "055": "bishop of llandaff",
    "056": "gaura", "057": "geranium", "058": "orange dahlia", "059": "tiger lily",
    "060": "pink-yellow dahlia", "061": "cautleya spicata", "062": "japanese anemone",
    "063": "black-eyed susan", "064": "silverbush", "065": "californian poppy", "066": "osteospermum",
    "067": "spring crocus", "068": "bearded iris", "069": "windflower", "070": "moon orchid",
    "071": "tree poppy", "072": "gazania", "073": "azalea", "074": "water lily",
    "075": "rose", "076": "thorn apple", "077": "morning glory", "078": "passion flower",
    "079": "lotus", "080": "toad lily", "081": "bird of paradise", "082": "anthurium",
    "083": "frangipani", "084": "clematis", "085": "hibiscus", "086": "columbine",
    "087": "desert-rose", "088": "tree mallow", "089": "magnolia", "090": "cyclamen",
    "091": "watercress", "092": "monkshood", "093": "canna lily", "094": "hippeastrum",
    "095": "bee balm", "096": "ball moss", "097": "foxglove", "098": "bougainvillea",
    "099": "camellia", "100": "mallow", "101": "mexican petunia", "102": "bromelia"
}


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

    # Define files to download and their local names
    download_files = [
        (FLOWER_URL, "102flowers.tgz"),
        (LABELS_URL, "imagelabels.mat"),
        (SETID_URL, "setid.mat")
    ]

    for url, filename in download_files:
        try:
            logging.info(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filename)
            logging.info(f"Successfully downloaded {filename}.")
        except urllib.error.URLError as e:
            logging.error(f"❌ Failed to download {filename} from {url}: {e}")
            raise # Re-raise to stop the process if essential download fails
        except Exception as e:
            logging.error(f"❌ An unexpected error occurred while downloading {filename}: {e}")
            raise

    try:
        logging.info("Extracting 102flowers.tgz...")
        with tarfile.open("102flowers.tgz") as tar:
            tar.extractall()
        logging.info("Successfully extracted 102flowers.tgz.")
    except tarfile.ReadError as e:
        logging.error(f"❌ Failed to extract 102flowers.tgz: {e}")
        raise
    except Exception as e:
        logging.error(f"❌ An unexpected error occurred while extracting 102flowers.tgz: {e}")
        raise

    logging.info("✅ Oxford 102 Flowers dataset downloaded and extracted.")

def prepare_oxford_splits(labels, train_ids, val_ids, image_dir="jpg"):
    """Organizes Oxford dataset into train and validation directories based on numeric labels."""
    def _prepare_split_dir(image_ids, target_dir):
        os.makedirs(target_dir, exist_ok=True)
        for i in image_ids:
            # Labels are 1-indexed in .mat file, convert to 0-indexed for folder names
            # and ensure 3-digit format (e.g., 1 -> "001")
            label_code = f"{labels[i - 1]:03d}"
            label_dir = os.path.join(target_dir, label_code)
            os.makedirs(label_dir, exist_ok=True)
            src = os.path.join(image_dir, f"image_{i:05d}.jpg")
            dst = os.path.join(label_dir, f"image_{i:05d}.jpg")
            if os.path.exists(src):
                shutil.copy(src, dst)
            else:
                logging.warning(f"Image not found during split preparation: {src}")

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
            # This is a heuristic to guess if it's a valid dataset structure
            if any(os.path.isdir(os.path.join(USER_DATA_DIR, d)) for d in os.listdir(USER_DATA_DIR)):
                logging.info(f"✅ Found user-uploaded data in {len(os.listdir(USER_DATA_DIR))} folders.")
            else:
                logging.warning("ℹ️ Uploaded data extracted but no class subdirectories found. Ensure it's class-wise organized.")
        else:
            logging.info("ℹ️ No images found in uploaded data or zip was empty.")

    except requests.exceptions.RequestException as e:
        logging.warning(f"❌ Failed to download uploaded data (network/HTTP error): {e}")
    except zipfile.BadZipFile:
        logging.warning("❌ Downloaded file is not a valid ZIP archive.")
    except Exception as e:
        logging.warning(f"❌ An unexpected error occurred during user data download: {e}")

def get_merged_datasets():
    """
    Loads and merges Oxford and user-uploaded datasets.
    Maps Oxford's numeric folder names (e.g., "001") to common names.
    """
    logging.info("📂 Loading datasets...")

    # Load Oxford datasets (classes will be like ['001', '002', ...])
    oxford_train_dataset = datasets.ImageFolder(OXFORD_TRAIN_DIR, transform=train_transform)
    oxford_val_dataset = datasets.ImageFolder(OXFORD_VAL_DIR, transform=val_transform)

    # Convert Oxford's numeric class names (e.g., "001") to common names
    # This list will contain the common names for all classes from Oxford
    # The order of these names will dictate the final class indexing if user data doesn't introduce them
    oxford_common_names = [OXFORD_CODE_TO_NAME_MAP.get(code, code) for code in oxford_train_dataset.classes]

    # Initialize combined unique classes with Oxford's common names
    all_unique_common_names = set(oxford_common_names)

    # Initialize merged datasets
    merged_train_dataset = oxford_train_dataset
    merged_val_dataset = oxford_val_dataset # Oxford's dedicated validation set

    if os.path.exists(USER_DATA_DIR) and os.listdir(USER_DATA_DIR):
        try:
            # Load user dataset (classes are assumed to be common names directly)
            user_dataset = datasets.ImageFolder(USER_DATA_DIR, transform=train_transform)
            if user_dataset:
                logging.info(f"Adding {len(user_dataset)} images from user-uploaded data with {len(user_dataset.classes)} classes.")
                # Update the set of all unique common names with user-provided class names
                all_unique_common_names.update(user_dataset.classes)

                # Concatenate Oxford training data with user data
                merged_train_dataset = ConcatDataset([oxford_train_dataset, user_dataset])
            else:
                logging.warning("User dataset is empty after loading.")
        except Exception as e:
            logging.warning(f"Could not load user dataset from '{USER_DATA_DIR}': {e}. Proceeding without it.")

    logging.info(f"✅ Total unique common names for training: {len(all_unique_common_names)}")

    # Sort the unique common names alphabetically to ensure consistent indexing
    # This sorted list will define the final 0-indexed mapping for the model's output
    final_sorted_common_names = sorted(list(all_unique_common_names))

    # --- Re-map dataset targets to unified indices ---
    # This is a crucial step to ensure the model outputs match the unified indices.
    # We need to map the original string labels (e.g., "001", "rose") to the new
    # numerical indices (0, 1, 2...) based on `final_sorted_common_names`.

    # Create a mapping from old class strings (folder names) to new unified indices (0, 1, 2...)
    # This maps '001' -> 'pink primrose' -> its new unified index
    # And 'rose' -> its new unified index
    class_string_to_unified_idx = {name: i for i, name in enumerate(final_sorted_common_names)}

    def remap_targets(dataset, is_oxford):
        # Create a new dataset that remaps the target labels
        remapped_samples = []
        for img_path, original_idx in dataset.samples:
            original_class_string = dataset.classes[original_idx]
            if is_oxford:
                # Map '001' to 'pink primrose', then 'pink primrose' to its new unified index
                common_name = OXFORD_CODE_TO_NAME_MAP.get(original_class_string, original_class_string)
            else:
                # User data class strings are already common names
                common_name = original_class_string

            unified_idx = class_string_to_unified_idx.get(common_name)
            if unified_idx is None:
                logging.warning(f"Class '{common_name}' not found in unified mapping. Skipping image: {img_path}")
                continue # Skip images with unmapped classes
            remapped_samples.append((img_path, unified_idx))
        dataset.samples = remapped_samples
        dataset.classes = final_sorted_common_names # Update dataset classes to be common names
        dataset.class_to_idx = class_string_to_unified_idx # Update class_to_idx

    logging.info("Remapping dataset labels to unified common name indices...")
    remap_targets(oxford_train_dataset, is_oxford=True)
    remap_targets(oxford_val_dataset, is_oxford=True)
    if 'user_dataset' in locals() and user_dataset: # Check if user_dataset was loaded
        remap_targets(user_dataset, is_oxford=False)

    # Recreate the merged_train_dataset with the remapped individual datasets
    # This is critical after the in-place remapping of `samples` in ImageFolder.
    if 'user_dataset' in locals() and user_dataset:
        final_train_source_dataset = ConcatDataset([oxford_train_dataset, user_dataset])
    else:
        final_train_source_dataset = oxford_train_dataset


    # Split the *remapped* merged_train_dataset into training and validation subsets
    train_size = int((1 - VALIDATION_SPLIT_RATIO) * len(final_train_source_dataset))
    val_size = len(final_train_source_dataset) - train_size
    train_subset, val_subset = random_split(final_train_source_dataset, [train_size, val_size])

    # Combine the validation split from the merged data AND the Oxford dedicated validation set
    # This ensures comprehensive validation
    final_val_dataset_combined = ConcatDataset([val_subset, oxford_val_dataset])

    final_train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() // 2 or 1)
    final_val_loader = DataLoader(final_val_dataset_combined, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() // 2 or 1)

    logging.info(f"Training loader size: {len(train_subset)} images, Validation loader size: {len(final_val_dataset_combined)} images.")
    return final_train_loader, final_val_loader, final_sorted_common_names, len(final_sorted_common_names)


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

# ... (code before train_model)

def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer):
    """Trains the model."""
    logging.info("📚 Starting model training...")
    best_val_loss = float('inf')  # Track best validation loss for checkpointing

    # This line needs to be consistently indented, usually 4 spaces from the 'def'
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # This 'for' loop should also be indented 4 spaces from the 'def'
    for epoch in range(num_epochs):
        # All lines inside the 'for' loop should be indented 8 spaces
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

            if batch_idx % 50 == 0:
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

        # Save the best model weights based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model_weights.pth")
            logging.info(f"Saving best model weights with validation loss: {best_val_loss:.4f}")

    logging.info("✅ Training complete.")

    # Load the best weights back into the model before scripting
    if os.path.exists("best_model_weights.pth"):
        model.load_state_dict(torch.load("best_model_weights.pth"))
        logging.info("Loaded best model weights for final saving.")

    return model
