cat > app_web.py << 'EOF'
# app_web.py
import streamlit as st
from PIL import Image
from app.inference import get_model
import os

st.set_page_config(
    page_title="AI-MedVision - Breast Cancer Detection",
    page_icon="🎗️",
    layout="centered"
)

# Custom CSS for pink theme with better readability
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
    /* Fix text colors */
    .stMarkdown, .stText, .stMetric, .stProgress > div > div {
        color: #1a1a1a !important;
    }
    /* Fix progress text */
    .stProgress > div > div > div {
        color: #1a1a1a !important;
    }
    /* Fix metric labels */
    [data-testid="stMetricLabel"] {
        color: #1a1a1a !important;
    }
    [data-testid="stMetricValue"] {
        color: #1a1a1a !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎗️ AI-MedVision")
st.markdown("### Breast Cancer Detection from Mammogram/Histopathology Images")

# Load model
@st.cache_resource
def load_model():
    return get_model()

try:
    model = load_model()
    # Get actual classes from model
    classes = model.classes
    st.success(f"✅ Model loaded successfully! Detects: {', '.join(classes)}")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Upload
uploaded_file = st.file_uploader(
    "Upload Breast Tissue Image",
    type=['jpg', 'jpeg', 'png', 'tiff'],
    help="Upload a mammogram or histopathology image for breast cancer detection"
)

if uploaded_file:
    # Display image
    image = Image.open(uploaded_file).convert('RGB')
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Predict button
    if st.button("🔍 Predict", use_container_width=True, type="primary"):
        with st.spinner("Analyzing image with AI..."):
            try:
                result = model.predict(image)
                
                # Get the label and confidence
                label = result['label']
                confidence = result['confidence'] * 100
                is_cancer = label.upper() == 'MALIGNANT'
                
                # Display results with big, readable text
                st.markdown("---")
                st.markdown("## 📊 Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Diagnosis",
                        label,
                        delta="⚠️ CANCER DETECTED" if is_cancer else "✅ CLEAR"
                    )
                
                with col2:
                    st.metric(
                        "Confidence",
                        f"{confidence:.1f}%"
                    )
                
                # Risk assessment with clear colors
                if is_cancer:
                    st.error(f"⚠️ **HIGH RISK** - Malignant detected with {confidence:.1f}% confidence. Please consult a doctor immediately.")
                else:
                    st.success(f"✅ **LOW RISK** - Benign with {confidence:.1f}% confidence. Continue regular screening.")
                
                # Probability bars with better visibility
                st.markdown("### 📊 Class Probabilities")
                for cls, prob in result['probs'].items():
                    prob_percent = prob * 100
                    # Color coding
                    if cls.upper() == 'MALIGNANT':
                        color = "#ff4444"  # Red for malignant
                        emoji = "⚠️"
                    else:
                        color = "#4CAF50"  # Green for benign
                        emoji = "✅"
                    
                    # Display with custom bar
                    st.markdown(f"""
                        <div style="margin: 10px 0;">
                            <div style="display: flex; justify-content: space-between; color: #1a1a1a; font-weight: bold;">
                                <span>{emoji} {cls}</span>
                                <span>{prob_percent:.1f}%</span>
                            </div>
                            <div style="width: 100%; height: 25px; background: #f0f0f0; border-radius: 12px; overflow: hidden; border: 1px solid #ddd;">
                                <div style="width: {prob_percent}%; height: 100%; background: {color}; border-radius: 12px; transition: width 0.8s ease;"></div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Additional info
                st.markdown("---")
                st.markdown("### 💡 What this means:")
                if is_cancer:
                    st.markdown("""
                        - 🏥 **Action Required**: Please schedule an appointment with an oncologist
                        - 📋 **Follow-up**: Additional tests may be recommended
                        - 💊 **Treatment**: Early detection improves treatment outcomes
                    """)
                else:
                    st.markdown("""
                        - 📋 **Continue Screening**: Regular check-ups are important
                        - 🏥 **Monitor**: Report any changes to your doctor
                        - ✅ **Good News**: No immediate action required
                    """)
                
                st.info("⚕️ **Disclaimer**: This AI tool is for educational purposes only. Always consult a qualified healthcare professional for medical decisions.")
                    
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                st.error("Please make sure the image is a valid breast tissue/mammogram image.")
EOF
