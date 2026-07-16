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

def evaluate_model(weights_path="runs/crop_weed_train/weights/best.pt", data_yaml="dataset/data.yaml"):
    """
    Evaluates the trained YOLOv8 model on the test dataset split.
    
    Args:
        weights_path (str): Relative path to the trained 'best.pt' weights.
        data_yaml (str): Relative path to the dataset configuration file.
    """
    # Verify that the weights exist before attempting evaluation
    if not os.path.exists(weights_path):
        logging.error(f"Trained weights not found at: {weights_path}. Please run train.py first.")
        return

    if not os.path.exists(data_yaml):
        logging.error(f"Dataset YAML not found at: {data_yaml}")
        return

    logging.info(f"Loading trained model from {weights_path}...")
    try:
        # Load the custom trained model
        model = YOLO(weights_path)
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return

    logging.info("Starting evaluation on the TEST dataset...")
    try:
        # Evaluate on the 'test' split. 
        # By default, model.val() evaluates on 'val', so we explicitly pass split='test'.
        # YOLOv8 automatically calculates Precision, Recall, mAP50, mAP50-95.
        # It also automatically generates and saves the Confusion Matrix and F1-score plots.
        metrics = model.val(
            data=data_yaml,
            split="test",
            project="runs",           # Base directory for saving runs
            name="crop_weed_eval",    # Specific folder name for this evaluation
            exist_ok=True             # Overwrite if exists
        )
        
        # Extract and compute specific metrics
        precision = metrics.box.p.mean()  # Mean precision across classes
        recall = metrics.box.r.mean()     # Mean recall across classes
        
        # F1 Score formula: 2 * (Precision * Recall) / (Precision + Recall)
        # Avoid division by zero
        f1_score = 0.0
        if (precision + recall) > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)

        map50 = metrics.box.map50
        map50_95 = metrics.box.map

        print("\n" + "="*40)
        print("TEST DATASET EVALUATION RESULTS")
        print("="*40)
        print(f"Mean Precision: {precision:.4f}")
        print(f"Mean Recall:    {recall:.4f}")
        print(f"F1 Score:       {f1_score:.4f}")
        print(f"mAP@50:         {map50:.4f}")
        print(f"mAP@50-95:      {map50_95:.4f}")
        print("="*40)
        
        logging.info("Evaluation complete. Plots (Confusion Matrix, F1-Curve, etc.) are saved in: runs/crop_weed_eval/")

    except Exception as e:
        logging.error(f"An error occurred during evaluation: {e}")

if __name__ == "__main__":
    # Define relative paths
    WEIGHTS_FILE = os.path.join("runs", "crop_weed_train", "weights", "best.pt")
    YAML_FILE = os.path.join("dataset", "data.yaml")
    
    evaluate_model(weights_path=WEIGHTS_FILE, data_yaml=YAML_FILE)
