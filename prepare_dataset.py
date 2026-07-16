import shutil
import random
from pathlib import Path
from PIL import Image

def verify_image(image_path):
    """
    Verifies if an image can be opened and is not corrupted.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def setup_directories(base_dir):
    """
    Creates the required YOLO directory structure.
    """
    dirs = [
        base_dir / 'images' / 'train',
        base_dir / 'images' / 'val',
        base_dir / 'images' / 'test',
        base_dir / 'labels' / 'train',
        base_dir / 'labels' / 'val',
        base_dir / 'labels' / 'test',
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return base_dir

def create_yaml(base_dir):
    """
    Creates the data.yaml file for YOLOv8.
    """
    yaml_content = f"""train: {base_dir.resolve() / 'images' / 'train'}
val: {base_dir.resolve() / 'images' / 'val'}
test: {base_dir.resolve() / 'images' / 'test'}

nc: 2
names: ['crop', 'weed']
"""
    yaml_path = base_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

def prepare_dataset(source_data_dir, output_dir, split_ratio=(0.7, 0.2, 0.1)):
    """
    Main function to parse, verify, split, and move the dataset.
    """
    source_path = Path(source_data_dir)
    dest_path = Path(output_dir)
    
    # 1. Setup directory structure and data.yaml
    setup_directories(dest_path)
    create_yaml(dest_path)
    
    # 2. Gather all images
    all_images = list(source_path.glob('*.jpeg')) + list(source_path.glob('*.jpg'))
    
    valid_pairs = []
    skipped_files = 0
    
    print("Verifying files...")
    # 3. Verify each image and check for labels
    for img_path in all_images:
        label_path = img_path.with_suffix('.txt')
        
        # Check if label exists
        if not label_path.exists():
            skipped_files += 1
            continue
            
        # Check if image is corrupted
        if not verify_image(img_path):
            skipped_files += 1
            continue
            
        valid_pairs.append((img_path, label_path))
        
    total_valid = len(valid_pairs)
    
    # 4. Shuffle and split
    random.seed(42)  # For reproducibility
    random.shuffle(valid_pairs)
    
    train_end = int(total_valid * split_ratio[0])
    val_end = train_end + int(total_valid * split_ratio[1])
    
    splits = {
        'train': valid_pairs[:train_end],
        'val': valid_pairs[train_end:val_end],
        'test': valid_pairs[val_end:]
    }
    
    print("\nCopying files to new structure...")
    # 5. Copy files to their new destinations
    for split_name, pairs in splits.items():
        for img_p, lbl_p in pairs:
            # Destination paths
            dest_img_p = dest_path / 'images' / split_name / img_p.name
            dest_lbl_p = dest_path / 'labels' / split_name / lbl_p.name
            
            # Copy files
            shutil.copy2(img_p, dest_img_p)
            shutil.copy2(lbl_p, dest_lbl_p)

    # 6. Print Statistics
    print("\n--- Preprocessing Statistics ---")
    print(f"Total images found initially: {len(all_images)}")
    print(f"Skipped files (missing labels/corrupted): {skipped_files}")
    print(f"Total valid image-label pairs: {total_valid}")
    print(f"Training images (70%): {len(splits['train'])}")
    print(f"Validation images (20%): {len(splits['val'])}")
    print(f"Test images (10%): {len(splits['test'])}")
    print("--------------------------------")
    print(f"\nDataset prepared at: {dest_path.resolve()}")

if __name__ == "__main__":
    # Define paths
    # The source is deeply nested based on our earlier analysis
    SOURCE_DIRECTORY = "Project5_Ag_Crop and weed detection/Project5_Ag_Crop and weed detection/Project5_Ag_Crop and weed detection/agri_data/data"
    OUTPUT_DIRECTORY = "dataset"
    
    # Run the script
    prepare_dataset(SOURCE_DIRECTORY, OUTPUT_DIRECTORY, split_ratio=(0.7, 0.2, 0.1))
