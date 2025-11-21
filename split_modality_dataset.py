import os
import shutil
import random

# Set random seed for reproducibility
random.seed(42)

# Source directories (your current folders)
SRC_DIR = r"D:/COVID-19_Radiography_Dataset/modality_dataset"  # <-- change this
XRAY_DIR = os.path.join(SRC_DIR, "xray")
CT_DIR = os.path.join(SRC_DIR, "ct")

# Destination directories
DEST_DIR = os.path.join(SRC_DIR, "split_dataset")
TRAIN_DIR = os.path.join(DEST_DIR, "train")
VAL_DIR = os.path.join(DEST_DIR, "val")

# Train/Val split ratio
TRAIN_RATIO = 0.8  # 80% for training, 20% for validation

def create_dir(path):
    """Create directory if not exists."""
    if not os.path.exists(path):
        os.makedirs(path)

def split_and_copy(src_folder, dest_train, dest_val):
    """Split images into train and val folders."""
    images = [f for f in os.listdir(src_folder)
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)

    split_idx = int(len(images) * TRAIN_RATIO)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # Copy training images
    for img in train_images:
        shutil.copy(os.path.join(src_folder, img),
                    os.path.join(dest_train, img))

    # Copy validation images
    for img in val_images:
        shutil.copy(os.path.join(src_folder, img),
                    os.path.join(dest_val, img))

    print(f"[{os.path.basename(src_folder)}] -> Train: {len(train_images)}, Val: {len(val_images)}")

def main():
    # Create output directories
    create_dir(TRAIN_DIR)
    create_dir(VAL_DIR)

    # Create class subfolders
    for folder in ["xray", "ct"]:
        create_dir(os.path.join(TRAIN_DIR, folder))
        create_dir(os.path.join(VAL_DIR, folder))

    # Split and copy
    split_and_copy(XRAY_DIR, os.path.join(TRAIN_DIR, "xray"), os.path.join(VAL_DIR, "xray"))
    split_and_copy(CT_DIR, os.path.join(TRAIN_DIR, "ct"), os.path.join(VAL_DIR, "ct"))

    print("\n✅ Dataset successfully split and organized!")
    print(f"Output saved to: {DEST_DIR}")

if __name__ == "__main__":
    main()
