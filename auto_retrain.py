# auto_retrain.py

# ... (rest of your imports and initial config)
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
import logging # <--- THIS LINE NEEDS TO BE HERE!
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
# ... (your DEVICE, MODEL_OUTPUT, etc. are fine)

# Oxford 102 Flowers dataset URLs
# ... (your URLs are fine)

# Training Hyperparameters
# ... (your hyperparameters are fine)

# ---------------------------
# OXFORD CODE TO COMMON NAME MAPPING
# Re-insert this entire dictionary.
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

# ... (your transforms and helper functions, which are mostly fine)

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
    oxford_common_names = [OXFORD_CODE_TO_NAME_MAP.get(code, code) for code in oxford_train_dataset.classes]

    # Initialize combined unique classes with Oxford's common names
    all_unique_common_names = set(oxford_common_names)

    # Initialize merged datasets
    merged_train_dataset = oxford_train_dataset # Will be concatenated later
    # merged_val_dataset = oxford_val_dataset # Not strictly needed here, re-evaluate later for final val loader

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
    # This maps '001' -> 'pink primrose', then 'pink primrose' -> its new unified index
    # And 'rose' -> its new unified index
    class_string_to_unified_idx = {name: i for i, name in enumerate(final_sorted_common_names)}

    def remap_targets(dataset, is_oxford):
        # Create a new list of (image_path, new_unified_index) tuples
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

        # Update the dataset's internal samples, classes, and class_to_idx
        # This modifies the ImageFolder instance in place.
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


# ... (rest of your build_model, train_model, save_model_and_mapping, MAIN EXECUTION sections)
