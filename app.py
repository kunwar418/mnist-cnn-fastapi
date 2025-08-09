from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Allow CORS so frontend can call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend URL in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model('mnist_cnn_model.h5')
print("Full model summary:")
model.summary()

print("Final output shape:", model.output_shape)

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('L')  # grayscale
    image = image.resize((28, 28))
    img_array = np.array(image).astype('float32') / 255.0
    img_array = 1.0 - img_array
    img_array = img_array.reshape(-1, 28, 28, 1)  # batch size 1
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = preprocess_image(image_bytes)
    preds = model.predict(img)
    print("Prediction output shape:", preds.shape)  # Debug print
    preds = preds[0]  # shape (10,)
    predicted_class = int(np.argmax(preds))
    confidence = float(np.max(preds))
    return {
        "predicted_digit": predicted_class,
        "confidence": confidence,
        "all_probabilities": preds.tolist()
    }
print(model.output_shape)
