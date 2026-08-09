import os
import uuid
from flask import Flask, render_template, request, jsonify
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd
from werkzeug.utils import secure_filename

try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None

# Application Configuration
class Config:
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'static/uploads')
    DISEASE_CSV = os.environ.get('DISEASE_CSV', 'data/disease_info.csv')
    TREATMENT_CSV = os.environ.get('TREATMENT_CSV', 'data/treatment_info.csv')
    MODEL_PATH = os.environ.get('MODEL_PATH', 'models/plant_disease_model_1_latest.pt')
    ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'}
    DEFAULT_PORT = 5001

# Initialize Flask App
app = Flask(__name__)
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

# Load Disease & Treatment Data
disease_info = pd.read_csv(Config.DISEASE_CSV, encoding='cp1252')
treatment_info = pd.read_csv(Config.TREATMENT_CSV, encoding='cp1252', keep_default_na=False)

# Initialize Model variable
model = None

def get_model():
    global model
    if model is not None:
        return model
    
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model weights file not found at '{Config.MODEL_PATH}'. "
            "Please download the weights and place them in the 'models/' directory."
        )
    
    try:
        net = CNN.CNN(39)
        net.load_state_dict(torch.load(Config.MODEL_PATH, map_location=torch.device('cpu')))
        net.eval()
        model = net
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load PyTorch model weights: {str(e)}")

def translate_text(text, lang):
    if GoogleTranslator is None:
        raise RuntimeError("Translation dependency is not installed")
    return GoogleTranslator(source='auto', target=lang).translate(text)

def get_disease_name_by_index(idx):
    try:
        rows = disease_info[disease_info['index'] == idx]
        if not rows.empty:
            val = rows.iloc[0]['disease_name']
            return str(val) if pd.notna(val) else f"Index {idx}"
    except Exception:
        pass
    return f"Index {idx}"

# Function to Predict Disease (Accepts image file path)
def prediction(image_path):
    try:
        # Load and normalize format (always force RGB representation)
        with Image.open(image_path) as image:
            image = image.convert('RGB')
            image = image.resize((224, 224))
            input_data = TF.to_tensor(image)
            input_data = input_data.view((-1, 3, 224, 224))
            
            net = get_model()
            with torch.no_grad():
                output = net(input_data)
                probabilities = torch.softmax(output, dim=1)[0]
                
                # Get top 3 predictions
                top_probs, top_indices = torch.topk(probabilities, k=3)
                
                pred_index = int(top_indices[0])
                confidence = float(top_probs[0])
                
                alternatives = []
                for i in range(1, 3):
                    alt_idx = int(top_indices[i])
                    alt_conf = float(top_probs[i])
                    # Display alternative if probability >= 5%
                    if alt_conf >= 0.05:
                        alt_name = get_disease_name_by_index(alt_idx)
                        alternatives.append({
                            "index": alt_idx,
                            "disease_name": alt_name,
                            "confidence": alt_conf
                        })
                        
            return pred_index, confidence, alternatives
    except Exception as e:
        raise ValueError(f"Model prediction inference failed: {str(e)}")

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')



@app.route('/submit', methods=['POST'])
def submit():
    if 'image' not in request.files:
        return "No image uploaded", 400

    image_file = request.files['image']
    if image_file.filename == '':
        return "No file selected", 400

    # Validate file extension
    original_filename = secure_filename(image_file.filename)
    file_extension = os.path.splitext(original_filename)[1].lower()
    if file_extension not in Config.ALLOWED_EXTENSIONS:
        return "Unsupported file type", 400

    # Save to a unique UUID-based filename inside uploads directory
    unique_filename = f"{uuid.uuid4().hex}{file_extension}"
    file_path = os.path.join(Config.UPLOAD_FOLDER, unique_filename)

    try:
        image_file.save(file_path)
    except Exception as e:
        return f"Failed to save upload: {str(e)}", 500

    # Predict Disease (under a safe try-finally runtime cleanup loop)
    try:
        # Validate that it is a valid image content using Pillow
        try:
            with Image.open(file_path) as img:
                img.verify()
        except Exception:
            return "Invalid or corrupted image file", 400

        pred, confidence, alternatives = prediction(file_path)
    except Exception as e:
        return f"Prediction failed: {str(e)}", 500
    finally:
        # Delete uploaded file immediately after inference to prevent disk leaks
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as cleanup_err:
                print(f"Failed to remove temp file {file_path}: {cleanup_err}")

    # Retrieve details from dataset by looking up where the 'index' column equals 'pred'
    disease_rows = disease_info[disease_info['index'] == pred]
    treatment_rows = treatment_info[treatment_info['index'] == pred]

    if disease_rows.empty or treatment_rows.empty:
        return "Disease details not found in dataset", 404

    # Extract fields safely
    title = disease_rows.iloc[0]['disease_name']
    description = disease_rows.iloc[0]['description']
    prevent = disease_rows.iloc[0]['Possible Steps']
    image_url = disease_rows.iloc[0]['image_url']
    active_ingredient = treatment_rows.iloc[0]['active_ingredient']
    treatment_type = treatment_rows.iloc[0]['treatment_type']
    purpose = treatment_rows.iloc[0]['purpose']
    verification_status = treatment_rows.iloc[0]['verification_status']

    # Handle float/NaN values from pandas safely
    title = title if pd.notna(title) else ""
    description = description if pd.notna(description) else ""
    prevent = prevent if pd.notna(prevent) else ""
    image_url = image_url if pd.notna(image_url) else ""
    active_ingredient = active_ingredient if pd.notna(active_ingredient) else ""
    treatment_type = treatment_type if pd.notna(treatment_type) else ""
    purpose = purpose if pd.notna(purpose) else ""
    verification_status = verification_status if pd.notna(verification_status) else ""

    # Classify confidence level based on conservative thresholding
    if confidence >= 0.85:
        confidence_level = "High"
    elif confidence >= 0.50:
        confidence_level = "Moderate"
    else:
        confidence_level = "Low"

    return render_template('submit.html', 
                           title=title, 
                           desc=description, 
                           prevent=prevent, 
                           image_url=image_url, 
                           pred=pred,
                           active_ingredient=active_ingredient,
                           treatment_type=treatment_type,
                           purpose=purpose,
                           verification_status=verification_status,
                           confidence=confidence,
                           confidence_level=confidence_level,
                           alternatives=alternatives)

@app.route('/translate', methods=['GET'])
def translate():
    lang = request.args.get('lang', 'en')
    pred_str = request.args.get('pred')

    if not pred_str:
        return jsonify({"error": "No disease index provided"}), 400

    try:
        pred = int(pred_str)
    except ValueError:
        return jsonify({"error": "Invalid disease index"}), 400

    # Retrieve details from dataset by looking up where the 'index' column equals 'pred'
    disease_rows = disease_info[disease_info['index'] == pred]
    treatment_rows = treatment_info[treatment_info['index'] == pred]

    if disease_rows.empty or treatment_rows.empty:
        return jsonify({"error": "Disease details not found in dataset"}), 404

    # Extract fields safely
    title = disease_rows.iloc[0]['disease_name']
    desc = disease_rows.iloc[0]['description']
    prevent = disease_rows.iloc[0]['Possible Steps']
    image_url = disease_rows.iloc[0]['image_url']
    active_ingredient = treatment_rows.iloc[0]['active_ingredient']
    treatment_type = treatment_rows.iloc[0]['treatment_type']
    purpose = treatment_rows.iloc[0]['purpose']
    verification_status = treatment_rows.iloc[0]['verification_status']

    # Handle float/NaN values from pandas safely
    title = title if pd.notna(title) else ""
    desc = desc if pd.notna(desc) else ""
    prevent = prevent if pd.notna(prevent) else ""
    image_url = image_url if pd.notna(image_url) else ""
    active_ingredient = active_ingredient if pd.notna(active_ingredient) else ""
    treatment_type = treatment_type if pd.notna(treatment_type) else ""
    purpose = purpose if pd.notna(purpose) else ""
    verification_status = verification_status if pd.notna(verification_status) else ""

    # ✅ Fix: Handle translation failures gracefully with original English fallback
    try:
        if lang != 'en':
            title = translate_text(title, lang)
            desc = translate_text(desc, lang)
            prevent = translate_text(prevent, lang)
            if active_ingredient and active_ingredient != "None" and active_ingredient != "None (Healthy Crop)":
                active_ingredient = translate_text(active_ingredient, lang)
            if treatment_type and treatment_type != "None":
                treatment_type = translate_text(treatment_type, lang)
            if purpose and purpose != "None":
                purpose = translate_text(purpose, lang)
    except Exception as e:
        print(f"Translation failed: {e}. Falling back to English content.")

    return jsonify({
        "title": title,
        "desc": desc,
        "prevent": prevent,
        "image_url": image_url,
        "active_ingredient": active_ingredient,
        "treatment_type": treatment_type,
        "purpose": purpose,
        "verification_status": verification_status
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", Config.DEFAULT_PORT))
    app.run(debug=True, port=port)
