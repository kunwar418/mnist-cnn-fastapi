# MNIST CNN FastAPI

A FastAPI-based web service for predicting handwritten digits using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

## 📌 Features
- CNN model trained on the MNIST dataset.
- FastAPI backend for serving predictions.
- Accepts image uploads and returns predicted digit with confidence score.
- Returns probabilities for all digits (0–9).

## 🛠 Tech Stack
- **Python**
- **TensorFlow / Keras**
- **FastAPI**
- **Uvicorn**
- **NumPy**
- **Pillow**

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/kunwar418/mnist-cnn-fastapi.git
   cd mnist-cnn-fastapi
```
```
python -m venv venv
```
```
venv\Scripts\activate
```
```
source venv/bin/activate
```
```
pip install -r requirements.txt
```
```
mnist-cnn-fastapi/
│
├── model/                  # Saved CNN model
├── main.py                 # FastAPI application
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
└── .gitignore              # Ignored files
```
```
uvicorn main:app --reload
```
```
http://127.0.0.1:8000
```
