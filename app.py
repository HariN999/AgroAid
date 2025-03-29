import os
from flask import Flask, render_template, request, jsonify
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd
from googletrans import Translator  # For translation

# Initialize Flask App
app = Flask(__name__)

# Ensure Upload Directory Exists
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Disease & Supplement Data
disease_info = pd.read_csv('data/disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('data/supplement_info.csv', encoding='cp1252')

# Load Model
model = CNN.CNN(39)
model.load_state_dict(torch.load("models/plant_disease_model_1_latest.pt"))
model.eval()

# Initialize Translator
translator = Translator()

# Temporary Storage for Last Prediction (Removes Need for `session`)
last_prediction = {}

# Function to Predict Disease
def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', 
                           supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']), 
                           disease=list(disease_info['disease_name']), 
                           buy=list(supplement_info['buy link']))

@app.route('/submit', methods=['POST'])
def submit():
    if 'image' not in request.files:
        return "No image uploaded", 400  # Handle missing file error

    image = request.files['image']
    if image.filename == '':
        return "No file selected", 400  # Handle empty filename

    # Save the uploaded image
    filename = image.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(file_path)

    # Predict Disease
    pred = prediction(file_path)

    # Get details from dataset
    title = disease_info['disease_name'][pred]
    description = disease_info['description'][pred]
    prevent = disease_info['Possible Steps'][pred]
    image_url = disease_info['image_url'][pred]
    supplement_name = supplement_info['supplement name'][pred]
    supplement_image_url = supplement_info['supplement image'][pred]
    supplement_buy_link = supplement_info['buy link'][pred]

    # Store results in memory (No Session Used)
    global last_prediction
    last_prediction = {
        "title": title,
        "desc": description,
        "prevent": prevent,
        "image_url": image_url
    }

    return render_template('submit.html', 
                           title=title, 
                           desc=description, 
                           prevent=prevent, 
                           image_url=image_url, 
                           pred=pred,
                           sname=supplement_name, 
                           simage=supplement_image_url, 
                           buy_link=supplement_buy_link)

@app.route('/translate', methods=['GET'])
def translate():
    lang = request.args.get('lang', 'en')

    # Get last detected disease details from memory
    if not last_prediction:
        return jsonify({"error": "No disease data available"}), 400

    title = last_prediction["title"]
    desc = last_prediction["desc"]
    prevent = last_prediction["prevent"]
    image_url = last_prediction["image_url"]

    # ✅ Fix: Handle translation failures
    try:
        if lang != 'en':
            title = translator.translate(title, dest=lang).text
            desc = translator.translate(desc, dest=lang).text
            prevent = translator.translate(prevent, dest=lang).text
    except Exception as e:
        print(f"Translation failed: {e}")
        return jsonify({"error": "Translation failed"}), 500

    return jsonify({
        "title": title,
        "desc": desc,
        "prevent": prevent,
        "image_url": image_url
    })

if __name__ == '__main__':
    app.run(debug=True)
