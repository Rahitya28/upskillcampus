import os
import glob
import logging
import torch
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_load(*args, **kwargs)
torch.load = _patched_load
from ultralytics import YOLO
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def predict_source(source_path, weights_path="runs/crop_weed_train/weights/best.pt", output_dir="outputs"):
    """
    Runs YOLOv8 inference on a single image or a directory of images.
    
    Args:
        source_path (str): Path to an image file or a folder of images.
        weights_path (str): Path to the trained YOLOv8 weights.
        output_dir (str): Directory where annotated images will be saved.
        
    Returns:
        dict: A dictionary containing prediction statistics.
    """
    if not os.path.exists(weights_path):
        logging.error(f"Weights not found at {weights_path}. Train the model first.")
        return None

    if not os.path.exists(source_path):
        logging.error(f"Source path {source_path} does not exist.")
        return None

    os.makedirs(output_dir, exist_ok=True)
    
    try:
        model = YOLO(weights_path)
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return None

    # Determine if source is a single file or directory
    if os.path.isdir(source_path):
        image_files = glob.glob(os.path.join(source_path, "*.[jJ][pP][gG]")) + \
                      glob.glob(os.path.join(source_path, "*.[jJ][pP][eE][gG]")) + \
                      glob.glob(os.path.join(source_path, "*.[pP][nN][gG]"))
        logging.info(f"Found {len(image_files)} images in directory.")
    else:
        image_files = [source_path]

    total_stats = {"images_processed": 0, "total_crops": 0, "total_weeds": 0}

    for img_path in image_files:
        logging.info(f"Processing: {img_path}")
        try:
            # Run inference
            results = model.predict(source=img_path, conf=0.25, save=False, verbose=False)
            
            for result in results:
                # Count crops (0) and weeds (1)
                boxes = result.boxes
                cls_indices = boxes.cls.tolist()
                
                crops = cls_indices.count(0.0)
                weeds = cls_indices.count(1.0)
                
                total_stats["total_crops"] += crops
                total_stats["total_weeds"] += weeds
                total_stats["images_processed"] += 1
                
                # Get average confidence for this image
                confidences = boxes.conf.tolist()
                avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
                
                # Generate annotated image array
                annotated_frame = result.plot()
                
                # Save output image
                filename = os.path.basename(img_path)
                out_path = os.path.join(output_dir, f"pred_{filename}")
                cv2.imwrite(out_path, annotated_frame)
                
                logging.info(f"Saved: {out_path} | Crops: {crops}, Weeds: {weeds}, Avg Conf: {avg_conf:.2f}")
                
        except Exception as e:
            logging.error(f"Error processing {img_path}: {e}")

    logging.info("\n--- Overall Prediction Summary ---")
    for k, v in total_stats.items():
        logging.info(f"{k}: {v}")
    
    return total_stats

if __name__ == "__main__":
    # Test on a few images from the test split
    TEST_SOURCE = "dataset/images/test"
    predict_source(TEST_SOURCE)
