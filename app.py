import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress TensorFlow oneDNN warnings

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from werkzeug.utils import secure_filename
import gdown
import glob
import time
import gc

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# Define Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Original Min-Max Values for Denormalization
density_min = 359.0
density_max = 3350.0
cv_min = 17.0
cv_max = 109.0
thk_min = 354.0
thk_max = 699.0

# Denormalization Function
def denormalize(values):
    denorm_values = []
    denorm_values.append(values[0] * (density_max - density_min) + density_min)
    denorm_values.append(values[1] * (cv_max - cv_min) + cv_min)
    denorm_values.append(values[2] * (thk_max - thk_min) + thk_min)
    return denorm_values

class CLAHETransform:
    def __call__(self, img):
        if len(img.shape) == 2:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)
        return img

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def load_model(model_url, model_path="/opt/render/project/src/model_storage/best_hcec_hybrid_model_weights_only.pth"):
    from hybrid_model import HybridHCECModel
    if not os.path.exists(model_path):
        gdown.download(model_url, model_path, quiet=False)
    model = HybridHCECModel().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    if device.type == "cpu":
        model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    model.eval()
    gc.collect()  # Free memory after loading
    return model

# Model URL from Google Drive
MODEL_URL = os.getenv("MODEL_URL", "https://drive.google.com/uc?export=download&id=1qLe4mGHlCDZvKfAPQoOpvJUVT9OLW6p0")
model = None  # Lazy-load model

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Error loading image")
    image = cv2.resize(image, (224, 224))  # Resize early
    image = CLAHETransform()(image)
    image = np.stack([image] * 3, axis=-1)
    image = Image.fromarray(image)
    image = transform(image)
    image = image.unsqueeze(0).to(device)
    gc.collect()  # Free memory
    return image

def classify_prediction(cell_density, corneal_thickness, cv):
    suggested_action = ""
    surgery_type = ""

    if cell_density < 500:
        suggested_action = "Severe endothelial failure, urgent intervention"
        surgery_type = "Penetrating Keratoplasty (PK)"
    elif 500 <= cell_density <= 1000:
        suggested_action = "High risk of corneal decompensation, immediate monitoring required"
        surgery_type = "DMEK/DSAEK if worsening"
    elif 1000 <= cell_density <= 1500:
        if corneal_thickness < 450:
            suggested_action = "Thin cornea, high risk, possible intervention"
            surgery_type = "DMEK/DSAEK if worsening"
        elif 450 <= corneal_thickness <= 600:
            suggested_action = "Possible corneal edema, detailed evaluation"
            surgery_type = "Consider DSAEK if vision is affected"
        else:
            suggested_action = "Polymegathism detected, frequent monitoring needed"
            surgery_type = "DMEK/DSAEK if symptoms worsen"
    elif 1500 <= cell_density <= 2000:
        if corneal_thickness < 450:
            suggested_action = "Thin cornea, needs monitoring, avoid unnecessary surgery"
            surgery_type = "No immediate surgery, regular check-ups"
        elif 450 <= corneal_thickness <= 600 and cv > 40:
            suggested_action = "Borderline case, polymegathism present, monitor for progression"
            surgery_type = "DMEK/DSAEK if symptoms worsen"
        else:
            suggested_action = "Thick cornea, monitor for endothelial dysfunction"
            surgery_type = "Consider DSAEK if symptoms worsen"
    elif 2000 <= cell_density <= 2500:
        if corneal_thickness < 450:
            suggested_action = "Low endothelial reserve, routine check-ups advised"
            surgery_type = "No surgery unless symptoms worsen"
        elif 450 <= corneal_thickness <= 500:
            suggested_action = "Slight polymegathism, observe for changes"
            surgery_type = "Routine check-up"
        elif 450 <= corneal_thickness <= 500 and cv <= 40:
            suggested_action = "Borderline case, requires periodic evaluation"
            surgery_type = "Regular eye check-up"
        else:
            suggested_action = "Possible mild edema, observe over time"
            surgery_type = "Routine check-up"
    elif 2500 <= cell_density <= 3000:
        if corneal_thickness > 600:
            suggested_action = "Healthy cornea, routine check-ups advised"
            surgery_type = "No surgery needed"
        else:
            suggested_action = "Healthy cornea, no intervention required"
            surgery_type = "No surgery needed"
    else:
        suggested_action = "Very healthy cornea, no intervention required"
        surgery_type = "No surgery needed"

    surgery_required = "Yes" if "DMEK" in surgery_type or "DSAEK" in surgery_type or "Keratoplasty" in surgery_type else "No"

    return {
        "suggested_action": suggested_action,
        "surgery_type": surgery_type,
        "surgery_required": surgery_required
    }

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    global model
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        filename = secure_filename(file.filename)
        temp_path = os.path.join("/opt/render/project/src/model_storage/uploads", filename)
        os.makedirs("/opt/render/project/src/model_storage/uploads", exist_ok=True)
        file.save(temp_path)
        if model is None:
            model = load_model(MODEL_URL)  # Lazy-load model
        image = preprocess_image(temp_path)
        with torch.no_grad():
            regression_output = model(image)
            regression_output = regression_output.squeeze().cpu().numpy()
            denorm_output = denormalize(regression_output)
        classification = classify_prediction(
            denorm_output[0],
            denorm_output[2],
            denorm_output[1]
        )
        os.remove(temp_path)
        for old_file in glob.glob("/opt/render/project/src/model_storage/uploads/*"):
            if os.path.getmtime(old_file) < time.time() - 3600:
                os.remove(old_file)
        gc.collect()  # Free memory
        return jsonify({
            'cell_density': round(float(denorm_output[0]), 2),
            'cv': round(float(denorm_output[1]), 2),
            'corneal_thickness': round(float(denorm_output[2]), 2),
            'suggested_action': classification['suggested_action'],
            'surgery_type': classification['surgery_type'],
            'surgery_required': classification['surgery_required']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 8001))  # Use Render's PORT
    app.run(debug=False, host='0.0.0.0', port=port)