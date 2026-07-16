# Crop and Weed Detection using YOLOv8

## Abstract
Precision agriculture heavily relies on the ability to distinguish between crops and weeds to optimize herbicide usage. This project leverages a state-of-the-art YOLOv8 object detection model to accurately detect and classify crops and weeds in real-time, helping to build smart, targeted spraying systems.

## Problem Statement
Traditional broadcasting of pesticides leads to excessive chemical use, soil degradation, and increased farming costs. A localized approach, where only identified weeds are sprayed, requires robust Computer Vision systems capable of handling varying lighting, scale, and occlusions in field images.

## Objectives
- Structure and preprocess an unstructured dataset of field images into YOLO format.
- Train a lightweight, fast, and accurate YOLOv8n model.
- Evaluate the model's metrics (Precision, Recall, mAP).
- Provide inference scripts for single/batch images.
- Build a user-friendly Flask Web Application for visualizing detections.

## Dataset
The dataset contains high-resolution 512x512 RGB images with YOLO-format annotations.
Classes:
- `0`: Crop
- `1`: Weed
The data is dynamically split into 70% Train, 20% Validation, and 10% Test.

## Installation

1. Clone or navigate to the repository.
2. Ensure you have Python 3.8+ installed.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Folder Structure
```
Crop-Weed-Detection/
│
├── dataset/                # Train, val, test images and labels, plus data.yaml
├── models/                 # Additional or exported models (Optional)
├── runs/                   # YOLOv8 training and evaluation outputs
├── static/                 # CSS and uploaded/output images for the Web App
│   ├── css/
│   ├── outputs/
│   └── uploads/
├── templates/              # HTML templates for the Web App
│   └── index.html
│
├── prepare_dataset.py      # Script to verify, split, and format data
├── train.py                # Script to train YOLOv8n
├── evaluate.py             # Script to evaluate model on test split
├── predict.py              # Script to run inference on folders/images
├── app.py                  # Flask application main execution script
└── requirements.txt        # Python dependencies
```

## How to Train
First, prepare the dataset:
```bash
python prepare_dataset.py
```
Then, execute the training script:
```bash
python train.py
```
The model will save weights inside `runs/crop_weed_train/weights/best.pt`.

## How to Evaluate
To test the model on the isolated 10% test dataset and generate metrics (F1, mAP, Confusion Matrix):
```bash
python evaluate.py
```

## How to Predict
You can use the inference script on a single image or directory:
```bash
python predict.py
```

## How to Run Flask Web App
Start the local server:
```bash
python app.py
```
Then open `http://127.0.0.1:5000` in your browser. Upload an image to see bounding boxes, weed/crop counts, and confidence scores dynamically rendered.

## Results
(After training is complete, the results such as mAP50 and F1-score will be generated automatically in the `runs/` directory).

## Future Scope
- Deploying the model to edge devices (NVIDIA Jetson, Raspberry Pi) for drone or tractor integrations.
- Exploring instance segmentation (YOLOv8-seg) for precise pixel-level mask delineation.
- Adding temporal tracking (DeepSORT) for video stream processing.
