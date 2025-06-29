import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf  # Use TensorFlow's interpreter, not tflite_runtime

st.title("🌿 Plant Disease Detection")

# Load TFLite model once with caching
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="quantized_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load class names (each line = 1 class)
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f if line.strip()]

uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="🖼 Uploaded Leaf Image", use_column_width=True)

    if st.button("🔍 Predict"):
        # Resize and normalize
        image_resized = image.resize((128, 128))
        img_array = np.array(image_resized, dtype=np.float32) / 255.0

        # Quantize input if needed
        scale, zero_point = input_details[0]['quantization']
        if scale > 0:
            img_array = img_array / scale + zero_point
            img_array = np.round(img_array).astype(np.uint8)

        img_array = np.expand_dims(img_array, axis=0)

        # Set input & run inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()

        # Get prediction
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        # Dequantize output if needed
        output_scale, output_zero_point = output_details[0]['quantization']
        if output_scale > 0:
            output_data = output_scale * (output_data.astype(np.float32) - output_zero_point)

        # Show top-5 predictions
        top5_idx = np.argsort(output_data)[-5:][::-1]
        st.write("### 🔢 Top 5 predictions:")
        for idx in top5_idx:
            st.write(f"**{class_names[idx]}** — {output_data[idx]:.4f}")

        # Final prediction
        prediction_idx = int(np.argmax(output_data))
        predicted_class = class_names[prediction_idx]
        confidence = output_data[prediction_idx] * 100

        st.success(f"🧪 Predicted: **{predicted_class}** with **{confidence:.2f}%** confidence")
