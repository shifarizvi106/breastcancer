# app_web.py
import streamlit as st
from PIL import Image
from app.inference import get_model
import os

st.set_page_config(
    page_title="AI-MedVision",
    page_icon="🩺",
    layout="centered"
)

# Custom CSS for pink theme
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #ffe4f0, #ffd2e6);
    }
    .main {
        background: white;
        border-radius: 20px;
        padding: 30px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🩺 AI-MedVision")
st.markdown("### Pneumonia Detection from Chest X-Ray")

# Load model
@st.cache_resource
def load_model():
    return get_model()

try:
    model = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Upload
uploaded_file = st.file_uploader(
    "Upload Chest X-Ray Image",
    type=['jpg', 'jpeg', 'png'],
    help="Upload a chest X-ray image for pneumonia detection"
)

if uploaded_file:
    # Display image
    image = Image.open(uploaded_file).convert('RGB')
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Predict button
    if st.button("🔍 Predict", use_container_width=True):
        with st.spinner("Analyzing image..."):
            try:
                result = model.predict(image)
                
                # Display results
                col1, col2 = st.columns(2)
                
                is_pneumonia = result['label'] == 'PNEUMONIA'
                
                with col1:
                    st.metric(
                        "Diagnosis",
                        result['label'],
                        delta="⚠️" if is_pneumonia else "✅"
                    )
                
                with col2:
                    st.metric(
                        "Confidence",
                        f"{result['confidence']*100:.1f}%"
                    )
                
                # Probability bars
                st.markdown("### 📊 Probabilities")
                for cls, prob in result['probs'].items():
                    color = "#ff7ab6" if cls == "PNEUMONIA" else "#4CAF50"
                    st.progress(prob, text=f"{cls}: {prob*100:.1f}%")
                    st.markdown(f'<div style="height:2px;background:{color};width:{prob*100}%;"></div>', unsafe_allow_html=True)
                
                # Risk assessment
                if is_pneumonia:
                    st.error("⚠️ **High Risk** - Please consult a doctor immediately")
                else:
                    st.success("✅ **Low Risk** - Continue regular screening")
                    
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
