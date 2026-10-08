cat > app_web.py << 'EOF'
# app_web.py
import streamlit as st
from PIL import Image
from app.inference import get_model

st.set_page_config(page_title="Breast Cancer Predictor", layout="wide")

st.markdown("""
<style>
.stApp { background: #f4dad4; }
h1, h2, h3, h4, h5, h6, p, label, div, span, button {
    color: #2e4060 !important;
    font-family: 'Times New Roman', Times, serif !important;
}
h1 { text-align: center; font-size: 2.8rem; margin-bottom: 0; }
.subtitle { text-align: center; font-size: 0.95rem; opacity: 0.75; margin-top: 0; margin-bottom: 30px; }
.block-container { padding-top: 1.5rem !important; }
.stButton > button {
    background: #dd8c96 !important;
    color: #f4dad4 !important;
    border: none !important;
    padding: 14px 50px !important;
    border-radius: 30px !important;
    font-size: 1.1rem !important;
    width: 100% !important;
}
.stButton > button:hover { background: #c97a85 !important; }
.stFileUploader > div > div {
    border: 2px dashed #dd8c96 !important;
    border-radius: 12px !important;
    background: rgba(255,255,255,0.5) !important;
    padding: 30px !important;
}
.stFileUploader label { color: #2e4060 !important; }
.stColumn {
    background: white !important;
    border-radius: 12px !important;
    padding: 25px !important;
    box-shadow: 0 2px 10px rgba(46,64,96,0.08) !important;
}
.stImage img { border-radius: 12px !important; }
.stProgress > div > div { background: #f4dad4 !important; border-radius: 20px !important; height: 8px !important; }
.stProgress > div > div > div { border-radius: 20px !important; }
hr { border-color: rgba(46,64,96,0.15) !important; margin: 20px 0 !important; }
.result-malignant { background: rgba(46,64,96,0.08); border-left: 5px solid #2e4060; padding: 20px; border-radius: 8px; margin: 10px 0; }
.result-benign { background: rgba(221,140,150,0.15); border-left: 5px solid #dd8c96; padding: 20px; border-radius: 8px; margin: 10px 0; }
.result-label { font-size: 1.8rem !important; font-weight: 700 !important; }
.result-conf { font-size: 1rem !important; opacity: 0.8; }
.disclaimer { text-align: center; font-size: 0.8rem; opacity: 0.6; margin-top: 30px; padding-top: 15px; border-top: 1px solid rgba(46,64,96,0.15); }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1>Breast Cancer Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Disclaimer: This AI tool is for educational purposes only. Always consult a qualified healthcare professional for medical decisions.</p>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return get_model()

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png', 'tiff'])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True)

with col2:
    if uploaded_file:
        if st.button("Predict", use_container_width=True):
            with st.spinner("Analyzing..."):
                result = model.predict(image)
                label = result['label']
                confidence = result['confidence'] * 100
                probs = result['probs']
                is_malignant = 'maglinant' in label.lower()

                st.markdown("---")
                st.markdown("#### Results")

                if is_malignant:
                    st.markdown(f'<div class="result-malignant"><div class="result-label">Malignant</div><div class="result-conf">Confidence: {confidence:.1f}%</div><p>Please consult a healthcare professional immediately.</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="result-benign"><div class="result-label">Benign</div><div class="result-conf">Confidence: {confidence:.1f}%</div><p>No malignancy detected. Continue regular screening.</p></div>', unsafe_allow_html=True)

                st.markdown("#### Confidence Breakdown")
                for cls, prob in probs.items():
                    name = "Malignant" if "maglinant" in cls.lower() else "Benign"
                    st.progress(prob, text=f"{name}: {prob*100:.1f}%")
    else:
        st.markdown('<p style="text-align:center; opacity:0.5; padding:40px 0;">Upload an image to begin analysis</p>', unsafe_allow_html=True)

st.markdown('<p class="disclaimer">Disclaimer: This AI tool is for educational purposes only. Always consult a qualified healthcare professional for medical decisions.</p>', unsafe_allow_html=True)
EOF
