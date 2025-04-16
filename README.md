
# 🌿 AgroAid: Multilingual Pest Diagnosis & Pesticide Recommendation

AgroAid is a web-based AI system that helps farmers identify crop diseases using images of plant leaves. It uses a Convolutional Neural Network (CNN) model trained to recognize various plant diseases and offers a clean UI for users to interact with the system.

---

## 🚀 Features

- Upload leaf images and detect diseases instantly
- Deep learning-based classification (CNN)
- Clean Flask-based web interface
- Deployable on local or cloud servers

---

## 🛠️ Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/HariN999/AgroAid.git
cd AgroAid
```

### 2. Set Up the Environment

Install required packages:

```bash
pip install -r requirements.txt
```

### 3. Run the App Locally

```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

### 4. Upload a Leaf Image

1. Use the "Choose File" button on the homepage to upload an image.
2. Click "Predict".
3. The app will show the predicted disease (if any).

---

## 🤖 Model Info

The model is defined in `CNN.py` and loaded in `app.py`. You can replace the existing model with your own trained `.h5` model if needed.

---

## 💡 How to Use This Repo

- **Run the app locally** using the above instructions.
- **Retrain or modify the model** in `CNN.py`.
- **Update the web interface** via `templates/` and `static/` folders.
---

## 🤝 Contributing

We welcome all contributions!

1. Fork the repo
2. Create a new branch (`git checkout -b feature/YourFeature`)
3. Make your changes
4. Commit (`git commit -m 'Add YourFeature'`)
5. Push to your fork (`git push origin feature/YourFeature`)
6. Create a Pull Request

---

## 📬 Contact

For questions, open an issue or connect via [LinkedIn](https://www.linkedin.com/in/narlakanti-hariharan).

