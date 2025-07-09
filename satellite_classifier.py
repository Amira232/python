import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import io
import base64

# Force demo mode (or set True if TensorFlow + model is available)
TENSORFLOW_AVAILABLE = False

# Page config
st.set_page_config(page_title="Satellite Image Classifier", page_icon="🛰️", layout="wide")

# Class info
CLASS_INFO = {
    'Cloudy': {'description': 'Cloudy areas.', 'color': '#87CEEB', 'icon': '☁️'},
    'Desert': {'description': 'Desert land.', 'color': '#F4A460', 'icon': '🏜️'},
    'Green_Area': {'description': 'Vegetated zones.', 'color': '#228B22', 'icon': '🌳'},
    'Water': {'description': 'Water bodies.', 'color': '#4682B4', 'icon': '💧'}
}

@st.cache_resource
def load_classification_model():
    if not TENSORFLOW_AVAILABLE:
        return "demo_model"
    try:
        from tensorflow.keras.models import load_model
        return load_model('Modelenv.v1.h5')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(img):
    if not TENSORFLOW_AVAILABLE:
        return np.random.rand(1, 255, 255, 3)
    img = img.convert('RGB')
    img = img.resize((255, 255))
    from tensorflow.keras.preprocessing import image
    img_array = image.img_to_array(img)
    return np.expand_dims(img_array, axis=0) / 255.0

def predict_image(model, img_array):
    class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']
    if not TENSORFLOW_AVAILABLE or model == "demo_model":
        np.random.seed(42)
        prediction = np.random.dirichlet(np.ones(4) * 2).reshape(1, -1)
    else:
        prediction = model.predict(img_array)
    idx = np.argmax(prediction[0])
    predicted_class = class_names[idx]
    confidence = prediction[0][idx]
    probabilities = {class_names[i]: float(prediction[0][i]) for i in range(4)}
    return predicted_class, confidence, probabilities

# UI starts
st.title("🛰️ Satellite Image Classifier")
st.markdown("📤 Upload a satellite image to classify as Cloudy, Desert, Green_Area, or Water.")

# Sidebar
st.sidebar.title("📋 Instructions")
st.sidebar.info("1. Upload an image\n2. View prediction\n3. Check confidence")

model = load_classification_model()
if model is None:
    st.stop()

uploaded_file = st.file_uploader("Choose an image (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image_pil = Image.open(uploaded_file)
    st.image(image_pil, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Processing..."):
        img_array = preprocess_image(image_pil)
        predicted_class, confidence, probabilities = predict_image(model, img_array)

    # Prediction box
    icon = CLASS_INFO[predicted_class]['icon']
    st.success(f"**Prediction:** {icon} {predicted_class} ({confidence:.1%})")
    st.markdown(f"> {CLASS_INFO[predicted_class]['description']}")

    # Confidence bar chart using Streamlit
    st.markdown("### Confidence Scores")
    conf_df = pd.DataFrame.from_dict(probabilities, orient='index', columns=["Confidence"])
    st.bar_chart(conf_df)

    # Confidence table
    st.markdown("### Detailed Confidence Table")
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    table_df = pd.DataFrame({
        "Class": [f"{CLASS_INFO[c]['icon']} {c}" for c, _ in sorted_probs],
        "Confidence": [f"{v:.2%}" for _, v in sorted_probs]
    })
    st.dataframe(table_df, use_container_width=True)

    # Image Info
    st.markdown("### Image Info")
    st.write(f"**Size:** {image_pil.size}")
    st.write(f"**Mode:** {image_pil.mode}")
    st.write(f"**Format:** {image_pil.format or 'Unknown'}")

    # Top 2 prediction analysis
    top1, top2 = sorted_probs[:2]
    st.markdown("### Top Predictions")
    st.write(f"🥇 {top1[0]}: {top1[1]:.1%}")
    st.write(f"🥈 {top2[0]}: {top2[1]:.1%}")
    st.write(f"📊 Confidence gap: {top1[1] - top2[1]:.1%}")
