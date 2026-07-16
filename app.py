import os
import time
import uuid
from flask import Flask, render_template, request, url_for
import torch
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_load(*args, **kwargs)
torch.load = _patched_load
from ultralytics import YOLO
import cv2

app = Flask(__name__)

# Configure upload and output folders
UPLOAD_FOLDER = os.path.join('static', 'uploads')
OUTPUT_FOLDER = os.path.join('static', 'outputs')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load Model globally to prevent reloading on every request
MODEL_PATH = os.path.join("runs", "crop_weed_train", "weights", "best.pt")
try:
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        model = None
        print("WARNING: Model weights not found. Please train the model first.")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

@app.route('/', methods=['GET'])
def index():
    """Renders the homepage."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    """Handles image upload and performs YOLO inference."""
    if model is None:
        return render_template('index.html', error="Model is not loaded or trained yet.")

    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded.")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No file selected.")

    if file:
        try:
            # Generate unique filename to prevent clashes
            ext = file.filename.split('.')[-1]
            unique_id = str(uuid.uuid4())
            filename = f"{unique_id}.{ext}"
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Start timing
            start_time = time.time()

            # Run inference
            results = model.predict(source=filepath, conf=0.25, save=False)
            result = results[0]

            # Calculate metrics
            boxes = result.boxes
            cls_indices = boxes.cls.tolist()
            confidences = boxes.conf.tolist()
            
            crop_count = cls_indices.count(0.0)
            weed_count = cls_indices.count(1.0)
            avg_conf = (sum(confidences) / len(confidences)) * 100 if confidences else 0.0
            
            # Save annotated image
            annotated_frame = result.plot()
            out_filename = f"pred_{filename}"
            out_filepath = os.path.join(app.config['OUTPUT_FOLDER'], out_filename)
            cv2.imwrite(out_filepath, annotated_frame)
            
            detection_time = time.time() - start_time

            # Prepare URLs for rendering
            original_url = url_for('static', filename=f"uploads/{filename}")
            annotated_url = url_for('static', filename=f"outputs/{out_filename}")

            return render_template(
                'index.html',
                success=True,
                original_img=original_url,
                annotated_img=annotated_url,
                crop_count=crop_count,
                weed_count=weed_count,
                confidence=f"{avg_conf:.2f}%",
                time=f"{detection_time:.3f} seconds"
            )

        except Exception as e:
            return render_template('index.html', error=f"An error occurred during prediction: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000)
