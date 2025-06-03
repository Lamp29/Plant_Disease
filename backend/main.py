from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Keras model
model = tf.keras.models.load_model(r"C:\Users\PRIYA\Desktop\PlantDiseaseProject\plant_disease\backend\model_accurate_compact.keras")

# Load class names
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f if line.strip()]

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        img_bytes = await image.read()
        input_data = preprocess_image(img_bytes)

        # Use Keras model to predict
        output_data = model.predict(input_data)

        prediction_idx = int(np.argmax(output_data))
        predicted_class = class_names[prediction_idx]

        return {"prediction": predicted_class}
    except Exception as e:
        return {"error": str(e)}
