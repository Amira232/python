import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import io

# Try to import TensorFlow, fall back to demo mode if not available
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.warning("\u26a0\ufe0f TensorFlow not found. Running in demo mode with simulated predictions.")

# Page configuration
st.set_page_config(
    page_title="Satellite Image Classifier",
    page_icon="\ud83d\ude80",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Class information
CLASS_INFO = {
    'Cloudy': {'description': 'Areas covered by clouds.', 'color': '#87CEEB', 'icon': '☁️'},
    'Desert': {'description': 'Dry, arid regions.', 'color': '#F4A460', 'icon': '🏜️'},
    'Green_Area': {'description': 'Vegetated areas.', 'color': '#228B22', 'icon': '🌳'},
    'Water': {'description': 'Water bodies.', 'color': '#4682B4', 'icon': '💧'}
}

@st.cache_resource
def load_classification_model():
    if not TENSORFLOW_AVAILABLE:
        return "demo_model"
    try:
        return load_model('Modelenv.v1.h5')
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None

def preprocess_image(img):
    if not TENSORFLOW_AVAILABLE:
        return np.random.rand(1, 255, 255, 3)
    img = img.convert('RGB').resize((255, 255))
    img_array = image.img_to_array(img)
    return np.expand_dims(img_array, axis=0) / 255.0

def predict_image(model, img_array):
    class_names = list(CLASS_INFO.keys())
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

def main():
    st.title("\ud83d\ude80 Satellite Image Classifier")

    st.sidebar.header("\ud83d\udccb Instructions")
    st.sidebar.markdown("""
    1. Upload a satellite image
    2. The app will classify it as Cloudy, Desert, Green Area, or Water
    3. View prediction and confidence
    """)

    model = load_classification_model()
    if model is None:
        st.stop()

    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        image_pil = Image.open(uploaded_file)
        st.image(image_pil, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Classifying..."):
            img_array = preprocess_image(image_pil)
            predicted_class, confidence, probabilities = predict_image(model, img_array)

        icon = CLASS_INFO[predicted_class]['icon']
        desc = CLASS_INFO[predicted_class]['description']
        st.success(f"**Prediction:** {icon} {predicted_class} ({confidence:.1%})")
        st.caption(desc)

        st.subheader("\ud83d\udcca Confidence Scores")
        conf_df = pd.DataFrame.from_dict(probabilities, orient='index', columns=["Confidence"])
        st.bar_chart(conf_df)

        st.subheader("\ud83d\udcc8 Detailed Table")
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        table_df = pd.DataFrame({
            "Class": [f"{CLASS_INFO[c]['icon']} {c}" for c, _ in sorted_probs],
            "Confidence": [f"{v:.2%}" for _, v in sorted_probs]
        })
        st.dataframe(table_df, use_container_width=True)

        st.subheader("\ud83d\udcc4 Image Info")
        st.write(f"**Size:** {image_pil.size}")
        st.write(f"**Mode:** {image_pil.mode}")
        st.write(f"**Format:** {image_pil.format or 'Unknown'}")

if __name__ == "__main__":
    main()
