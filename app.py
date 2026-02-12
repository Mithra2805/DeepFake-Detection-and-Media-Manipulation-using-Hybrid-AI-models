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

st.set_page_config(page_title="Deepfake Detection System", layout="centered")
st.title("🛡 Deepfake Detection Using Hybrid AI")

st.markdown("""
This system analyzes uploaded media using multiple AI detection methods 
to identify potential deepfakes.
""")

uploaded_file = st.file_uploader(
    "Upload Image / Video / Audio",
    type=["jpg", "png", "jpeg", "mp4", "avi", "mov", "wav", "mp3", "ogg", "opus", "m4a", "aac"]  # ✅ Added ogg, opus, m4a, aac
)

if uploaded_file:
    # Save uploaded file temporarily with original extension
    file_extension = os.path.splitext(uploaded_file.name)[1]
    original_filename = uploaded_file.name  # Keep original filename
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp:
        temp.write(uploaded_file.read())
        path = temp.name

    results = []

    # Process based on file type
    if uploaded_file.type.startswith("image"):
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        with st.spinner("Analyzing image..."):
            # Pass both path and original filename
            label, conf = predict_image(path, original_filename)
            results.append((label, conf))
            
            st.subheader("📸 Image Analysis")
            st.write(f"**Detection:** {label}")
            st.write(f"**Confidence:** {conf}")

    elif uploaded_file.type.startswith("video"):
        st.video(uploaded_file)
        
        with st.spinner("Analyzing video..."):
            # Pass both path and original filename
            label, conf = predict_video(path, original_filename)
            results.append((label, conf))
            
            st.subheader("🎥 Video Analysis")
            st.write(f"**Detection:** {label}")
            st.write(f"**Confidence:** {conf}")

    elif uploaded_file.type.startswith("audio") or file_extension.lower() in ['.ogg', '.opus', '.m4a', '.aac']:  # ✅ Added fallback check
        st.audio(uploaded_file)
        
        with st.spinner("Analyzing audio..."):
            # Pass both path and original filename
            label, conf = predict_audio(path, original_filename)
            results.append((label, conf))
            
            st.subheader("🎵 Audio Analysis")
            st.write(f"**Detection:** {label}")
            st.write(f"**Confidence:** {conf}")

    # Hybrid fusion decision
    if results:
        final_label, final_conf = hybrid_decision(results)

        st.markdown("---")
        st.subheader("🔍 Final Detection Result")
        
        if final_label == "Deepfake":
            st.error(f"⚠️ **Result: {final_label}**")
            st.progress(final_conf)
            st.metric("Confidence Level", f"{final_conf * 100:.0f}%")
        else:
            st.success(f"✅ **Result: {final_label}**")
            st.progress(final_conf)
            st.metric("Confidence Level", f"{final_conf * 100:.0f}%")
        
        with st.expander("ℹ️ How does this work?"):
            st.write("""
            - **Image Analysis**: Checks EXIF metadata, dimensions, and filename patterns
            - **Video Analysis**: Examines frame count, FPS, and temporal consistency
            - **Audio Analysis**: Analyzes file properties and patterns
            - **Hybrid Fusion**: Combines all analyses with weighted voting
            
            **Note**: This is a demonstration system. For production use, 
            replace heuristics with trained deep learning models.
            """)
    
    # Clean up temp file
    try:
        os.unlink(path)
    except:
        pass

st.sidebar.header("About")
st.sidebar.info("""
**Deepfake Detection System v1.0**

Upload images, videos, or audio files to detect potential deepfakes 
using hybrid AI analysis.

**Supported Formats:**
- Images: JPG, PNG, JPEG
- Videos: MP4, AVI, MOV
- Audio: WAV, MP3, OGG, OPUS, M4A, AAC

**Pro Tip:** Name AI-generated files with keywords like 'ai', 'generated', or 'fake' for better detection.
""")

st.sidebar.header("Tips for Best Results")
st.sidebar.markdown("""
- Use clear, high-quality media
- Name files descriptively (e.g., 'ai_voice.mp3')
- Larger files provide more data for analysis
- Results are probabilistic, not definitive
""")