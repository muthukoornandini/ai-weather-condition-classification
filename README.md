# 🌦️ AI-Based Weather Condition Classification System

A deep learning project that classifies weather conditions from images using
**Transfer Learning (MobileNetV2)** built with **TensorFlow / Keras**.

The model predicts one of four weather classes from a photograph:

- ☁️ Cloudy
- 🌧️ Rain
- ☀️ Shine
- 🌅 Sunrise

---

## 📁 Project Structure

```
weather-classification/
│
├── data/
│   ├── raw/                 # Put your original downloaded dataset here (class-wise folders)
│   └── dataset/             # Auto-generated train/val/test split (created by prepare_dataset.py)
│
├── models/                  # Trained models get saved here (.h5 / .keras)
│
├── notebooks/
│   └── weather_classification_demo.ipynb   # Step-by-step notebook version
│
├── src/
│   ├── config.py             # All hyperparameters & paths in one place
│   ├── data_loader.py        # Data generators / augmentation pipeline
│   ├── model.py               # CNN & transfer-learning model architecture
│   ├── train.py               # Training script (run this to train the model)
│   ├── evaluate.py            # Evaluate on test set + confusion matrix + report
│   └── predict.py             # Predict weather condition for a single image (CLI)
│
├── utils/
│   └── prepare_dataset.py    # Splits raw class folders into train/val/test
│
├── app.py                    # Streamlit web app for interactive demo
├── requirements.txt
└── README.md
```

---

## 📦 1. Setup

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🖼️ 2. Get the Dataset

This project is designed around the popular **Multi-class Weather Dataset**
(Cloudy / Rain / Shine / Sunrise, ~1125 images), available on Kaggle:

> Search: *"Multi-class Weather Dataset for Image Classification"* on Kaggle

Download it and arrange it like this inside `data/raw/`:

```
data/raw/
├── cloudy/
│   ├── img1.jpg
│   └── ...
├── rain/
│   ├── img1.jpg
│   └── ...
├── shine/
│   ├── img1.jpg
│   └── ...
└── sunrise/
    ├── img1.jpg
    └── ...
```

> 💡 You can use **any** weather image dataset — just make sure images are
> organized into one folder per class name, and update `CLASS_NAMES` in
> `src/config.py` if your class names differ.

Then split it into train/val/test:

```bash
python utils/prepare_dataset.py
```

This creates `data/dataset/train`, `data/dataset/val`, and `data/dataset/test`
(80% / 10% / 10% split by default — configurable via CLI flags).

---

## 🏋️ 3. Train the Model

```bash
python src/train.py
```

What it does:
- Loads MobileNetV2 pretrained on ImageNet (transfer learning) as the backbone
- Freezes the backbone, trains a custom classification head
- Fine-tunes the top backbone layers for a few extra epochs
- Uses data augmentation (rotation, zoom, flips, shifts) to reduce overfitting
- Saves the best model checkpoint to `models/weather_model_best.h5`
- Saves the final model to `models/weather_model_final.h5`
- Plots & saves accuracy/loss curves to `models/training_history.png`

All hyperparameters (image size, batch size, epochs, learning rate, etc.) live
in `src/config.py` — edit them there.

---

## 📊 4. Evaluate

```bash
python src/evaluate.py
```

Prints accuracy, precision/recall/F1 per class, and saves a confusion matrix
image to `models/confusion_matrix.png`.

---

## 🔮 5. Predict on a Single Image

```bash
python src/predict.py --image path/to/photo.jpg
```

Example output:
```
Predicted class : Rain
Confidence      : 96.42%

Class probabilities:
  cloudy   :  1.20%
  rain     : 96.42%
  shine    :  0.88%
  sunrise  :  1.50%
```

---

## 🌐 6. Interactive Web Demo (Streamlit)

```bash
streamlit run app.py
```

Upload any weather photo in the browser and get an instant prediction with a
probability bar chart.

---

## 🧠 Model Architecture

```
Input (224x224x3)
      │
MobileNetV2 (ImageNet weights, frozen initially)
      │
GlobalAveragePooling2D
      │
Dense(256, relu) + Dropout(0.4)
      │
Dense(128, relu) + Dropout(0.3)
      │
Dense(num_classes, softmax)
```

Training strategy: **two-phase transfer learning**
1. Train only the new head with the backbone frozen.
2. Unfreeze the last N backbone layers and fine-tune at a lower learning rate.

---

## 🛠️ Tech Stack

- Python 3.9+
- TensorFlow / Keras
- NumPy, Matplotlib, scikit-learn
- Streamlit (demo UI)
- Pillow (image handling)

---

## 📈 Possible Extensions

- Add more classes (fog, snow, thunderstorm, hail)
- Swap MobileNetV2 for EfficientNet / ResNet50 for higher accuracy
- Deploy as a REST API with FastAPI / Flask
- Convert to TensorFlow Lite for mobile/edge deployment
- Add Grad-CAM visualization to explain predictions

---

## 📄 License

This project is provided for educational purposes. Feel free to modify and
reuse it for your own coursework, portfolio, or research projects.
