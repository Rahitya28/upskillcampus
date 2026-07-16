import os
import logging
import torch
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_load(*args, **kwargs)
torch.load = _patched_load
from ultralytics import YOLO

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_yolo(data_yaml_path="dataset/data.yaml", epochs=50, imgsz=512, batch_size=16):
    """
    Initializes and trains the YOLOv8 model for crop and weed detection.
    
    Args:
        data_yaml_path (str): Relative path to the dataset.yaml file.
        epochs (int): Number of epochs to train the model.
        imgsz (int): Image resolution (images are 512x512).
        batch_size (int): Batch size for training.
    """
    # Validate that the dataset configuration exists
    if not os.path.exists(data_yaml_path):
        logging.error(f"Dataset configuration file not found at: {data_yaml_path}")
        return

    logging.info("Initializing YOLOv8n model...")
    # Why YOLOv8n? 
    # YOLOv8n ('nano') is selected because it is the most lightweight model in the YOLOv8 family.
    # It trains very quickly, requires minimal computational resources, and provides incredibly fast 
    # inference speeds. This makes it ideal for real-time agricultural applications where edge devices 
    # (like embedded systems on tractors or drones) need to process images on the fly.
    try:
        model = YOLO("yolov8n.pt") 
    except Exception as e:
        logging.error(f"Failed to load YOLOv8n model: {e}")
        return

    logging.info(f"Starting training for {epochs} epochs...")
    
    # Train the model
    # YOLO automatically saves 'best.pt' and 'last.pt' in the specified project/name directory
    # (runs/detect/train/weights/ by default, but we customize it here).
    # It also automatically logs and displays Training Loss, Validation Loss, Precision, Recall, mAP50, and mAP50-95.
    try:
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            project="runs",         # Directory to save training runs
            name="crop_weed_train", # Name of the specific run
            exist_ok=True           # Overwrite if exists
        )
        logging.info("Training completed successfully.")
        
        # Display final metrics explicitly from the results dictionary
        metrics = results.results_dict
        print("\n" + "="*30)
        print("FINAL VALIDATION METRICS")
        print("="*30)
        # The exact keys depend on the ultralytics version, usually they look like:
        # 'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)'
        for key, value in metrics.items():
            if 'precision' in key.lower() or 'recall' in key.lower() or 'map' in key.lower():
                print(f"{key}: {value:.4f}")
        print("="*30)
        print("Weights saved at: runs/crop_weed_train/weights/best.pt and last.pt")

    except Exception as e:
        logging.error(f"An error occurred during training: {e}")

if __name__ == "__main__":
    # Define relative path to the YAML file
    YAML_PATH = os.path.join("dataset", "data.yaml")
    
    # Configurable epochs (can be changed here)
    EPOCHS = 50 
    
    train_yolo(data_yaml_path=YAML_PATH, epochs=EPOCHS, imgsz=512)
