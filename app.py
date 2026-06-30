import os
import tempfile
import streamlit as st
import numpy as np
try:
    import tensorflow as tf
except ImportError:
    tf = None

import preprocessing
# Force reload tracking: structural fake noise frequency boundaries successfully elevated
from utils.explainability import get_gradcam, overlay_heatmap

# Page Config
st.set_page_config(
    page_title="Deepfake Audio Detector",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E1E1E;
        margin-bottom: 0px;
    }
    .sub-top {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .pred-real {
        color: #2e7d32;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .pred-fake {
        color: #c62828;
        font-weight: bold;
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

def load_model_fresh_v4():
    """Load the Keras model dynamically from the filesystem."""
    model_path = "models/deepfake_audio_detector.h5"
    if os.path.exists(model_path) and tf is not None:
        return tf.keras.models.load_model(model_path)
    return None

def main():
    st.markdown("<div class='main-header'>🎙️ Deepfake Audio Detection System</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-top'>Upload an audio clip to detect AI-generated voice masking.</div>", unsafe_allow_html=True)

    # Sidebar: Model status & Settings
    with st.sidebar:
        st.header("⚙️ System Status")
        model = load_model_fresh_v4()
        if model:
            st.success("✅ Model loaded successfully.", icon="🧠")
            st.write("Ready for inference.")
        else:
            model_path = "models/deepfake_audio_detector.h5"
            st.error(f"❌ Model not loaded.", icon="⚠️")
            st.warning(f"Engine status: TensorFlow loaded? {tf is not None}. Model file found? {os.path.exists(model_path)}")
            st.info("Please run `python train.py` to generate the `.h5` model first.")
            if st.button("Reload System"):
                st.cache_resource.clear()
                st.rerun()
            st.stop()
            
        st.divider()
        use_explainability = st.checkbox("🔍 Enable Explainability (Grad-CAM Heatmap)", value=True, 
                                        help="Overlays a heatmap to show which parts of the spectrogram influenced the model's decision.")

    # Upload Section
    uploaded_file = st.file_uploader("Upload Audio (WAV/MP3)", type=["wav", "mp3"])
    
    # Optional Microphone feature (Bonus)
    st.markdown("**Or interact live**")
    audio_mic = st.audio_input("Record a voice clip directly")
    
    audio_source = audio_mic if audio_mic else uploaded_file

    if audio_source is not None:
        st.audio(audio_source)
        
        # Save uploaded file temporarily to process through librosa
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            tmp_audio.write(audio_source.getvalue())
            tmp_audio_path = tmp_audio.name
            
        st.markdown("### ⚙️ Analysis in Progress")
        
        with st.spinner("Extracting features and running model..."):
            try:
                # 1. Preprocess
                y, sr = preprocessing.load_audio(tmp_audio_path)
                spec_array = preprocessing.extract_mel_spectrogram(y, sr)  # Shape: (128, 128, 1)
                
                # 2. Predict
                # Add batch dimension
                input_tensor = np.expand_dims(spec_array, axis=0) 
                prediction = model.predict(input_tensor)[0]
                
                # Classes: 0 -> Real, 1 -> Fake 
                # (As defined in train.py)
                real_confidence = prediction[0]
                fake_confidence = prediction[1]
                
                label = "REAL AUDIO" if real_confidence > fake_confidence else "DEEPFAKE DETECTED"
                confidence = max(real_confidence, fake_confidence) * 100
                
                # 3. Layout the Results
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("#### 📊 Spectrogram Analysis")
                    fig = preprocessing.plot_spectrogram(y, sr, title="Mel-Spectrogram Features")
                    st.pyplot(fig)
                    
                    if use_explainability:
                        st.markdown("#### 🔍 Explainability Heatmap (Grad-CAM)")
                        # Remove the batch and channel dim for the overlay function
                        base_img = np.squeeze(spec_array)  
                        heatmap = get_gradcam(model, input_tensor)
                        overlay = overlay_heatmap(heatmap, base_img)
                        st.image(overlay, caption="Regions that strongly influenced the prediction", use_container_width=True)

                with col2:
                    st.markdown("#### 🎯 Classification Result")
                    result_class = 'pred-real' if real_confidence > fake_confidence else 'pred-fake'
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="{result_class}">{label}</div>
                            <h2 style='margin: 10px 0;'>{confidence:.1f}% Confidence</h2>
                            <p style='color: grey;'>Real: {real_confidence*100:.1f}% | Fake: {fake_confidence*100:.1f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.progress(float(fake_confidence), text="Deepfake Probability")
            except Exception as e:
                st.error(f"Error processing audio: {e}")
            finally:
                # Cleanup temp file
                if os.path.exists(tmp_audio_path):
                    os.remove(tmp_audio_path)

if __name__ == "__main__":
    main()
