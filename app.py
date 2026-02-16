"""
Streamlit Web Application for Hybrid Deepfake Detection System
Supports multi-modal analysis: Images, Videos, and Audio
"""

import streamlit as st
import tempfile
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.image_model import predict_image
from models.video_model import predict_video
from models.audio_model import predict_audio
from utils.fusion import hybrid_decision
from config.model_config import Config

# Page configuration
st.set_page_config(
    page_title="Hybrid Deepfake Detection System",
    page_icon="🔍",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        padding: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🛡️ Hybrid Deepfake Detection System</h1>', unsafe_allow_html=True)

st.markdown("""
This system uses **deep learning models** to detect deepfakes across multiple modalities:
- 🖼️ **Images**: EfficientNet-B4 CNN
- 🎥 **Videos**: Temporal frame analysis
- 🎵 **Audio**: Mel-spectrogram CNN

**Architecture**: Hybrid multi-modal fusion with weighted confidence aggregation
""")

# Display system info
with st.expander("ℹ️ System Information"):
    st.write(f"**Device**: {Config.DEVICE}")
    st.write(f"**Image Model**: EfficientNet-B4")
    st.write(f"**Image Size**: {Config.IMAGE_SIZE}x{Config.IMAGE_SIZE}")
    st.write(f"**Video Sample Rate**: Every {Config.VIDEO_SAMPLE_RATE} frames")
    st.write(f"**Audio Sample Rate**: {Config.AUDIO_SAMPLE_RATE} Hz")
    st.write(f"**Fusion Weights**: {Config.FUSION_WEIGHTS}")

# File uploader
uploaded_file = st.file_uploader(
    "Upload Image / Video / Audio for Deepfake Detection",
    type=["jpg", "png", "jpeg", "mp4", "avi", "mov", "wav", "mp3", "ogg", "opus", "m4a", "aac"]
)

if uploaded_file:
    # Save uploaded file temporarily
    file_extension = os.path.splitext(uploaded_file.name)[1]
    original_filename = uploaded_file.name
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp:
        temp.write(uploaded_file.read())
        path = temp.name

    results = []
    modalities = []

    # Process based on file type
    if uploaded_file.type.startswith("image"):
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        with st.spinner("🔍 Analyzing image with deep learning model..."):
            label, conf = predict_image(path, original_filename)
            results.append((label, conf))
            modalities.append('image')
            
            st.subheader("📸 Image Analysis Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Detection", label, delta=None)
            with col2:
                st.metric("Confidence", f"{conf:.2%}")

    elif uploaded_file.type.startswith("video"):
        st.video(uploaded_file)
        
        with st.spinner("🔍 Analyzing video frames with CNN model..."):
            label, conf = predict_video(path, original_filename)
            results.append((label, conf))
            modalities.append('video')
            
            st.subheader("🎥 Video Analysis Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Detection", label)
            with col2:
                st.metric("Confidence", f"{conf:.2%}")

    elif uploaded_file.type.startswith("audio") or file_extension.lower() in ['.ogg', '.opus', '.m4a', '.aac']:
        st.audio(uploaded_file)
        
        with st.spinner("🔍 Analyzing audio with mel-spectrogram CNN..."):
            label, conf = predict_audio(path, original_filename)
            results.append((label, conf))
            modalities.append('audio')
            
            st.subheader("🎵 Audio Analysis Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Detection", label)
            with col2:
                st.metric("Confidence", f"{conf:.2%}")

    # Multi-modal fusion
    if results:
        final_label, final_conf = hybrid_decision(results, modalities)

        st.markdown("---")
        st.subheader("🔬 Final Hybrid Detection Result")
        
        # Display result with color coding
        if final_label == "Deepfake":
            st.error(f"⚠️ **DETECTED: {final_label}**")
            color = "red"
        else:
            st.success(f"✅ **DETECTED: {final_label}**")
            color = "green"
        
        # Confidence visualization
        st.progress(final_conf)
        st.markdown(f'<div class="metric-card"><h3 style="color:{color};">Confidence: {final_conf:.1%}</h3></div>', 
                   unsafe_allow_html=True)
        
        # Technical details
        with st.expander("🔧 Technical Details"):
            st.write("**Detection Pipeline:**")
            for i, ((label, conf), mod) in enumerate(zip(results, modalities)):
                weight = Config.FUSION_WEIGHTS.get(mod, 0.0)
                st.write(f"{i+1}. {mod.title()} Model: {label} ({conf:.2%}) - Weight: {weight:.2%}")
            
            st.write("\n**Fusion Method:** Weighted confidence aggregation")
            st.write(f"**Final Decision:** {final_label} with {final_conf:.1%} confidence")
    
    # Clean up temp file
    try:
        os.unlink(path)
    except:
        pass

# Sidebar
st.sidebar.header("About")
st.sidebar.info("""
**Hybrid Deepfake Detection System**

A multi-modal deep learning system for detecting synthetic media manipulation.

**Models:**
- Image: EfficientNet-B4
- Video: Frame-level CNN + Temporal fusion
- Audio: Mel-spectrogram CNN

**Technology Stack:**
- PyTorch
- OpenCV
- Librosa
- Streamlit
""")

st.sidebar.header("Supported Formats")
st.sidebar.markdown("""
**Images:** JPG, PNG, JPEG  
**Videos:** MP4, AVI, MOV  
**Audio:** WAV, MP3, OGG, OPUS, M4A, AAC
""")

st.sidebar.header("How It Works")
st.sidebar.markdown("""
1. **Upload** media file
2. **Extract** features using deep learning
3. **Analyze** with trained CNN models
4. **Fuse** predictions from multiple modalities
5. **Output** final detection result
""")