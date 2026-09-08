# Deepfake Audio Detection System 🎙️

An AI-powered audio forensic system built using Convolutional Neural Networks (CNNs), Librosa, and Streamlit to detect synthetic AI-generated speech and voice deepfakes. Includes Grad-CAM explainability heatmaps to visualize frequency anomalies.

---

## 🌟 Features

- **Multi-Segment Audio Analysis**: Evaluates long recordings (speech, music, singing) via overlapping sliding windows rather than truncating, delivering reliable predictions.
- **Explainable AI (Grad-CAM)**: Generates class activation heatmaps overlaying the Mel-Spectrogram to highlight spectral and harmonic anomalies influencing the prediction.
- **Interactive Web Interface**: Streamlit UI with drag-and-drop audio uploading (`.wav`, `.mp3`) and live microphone recording.
- **Accurate Model**: CNN architecture trained on authentic speech and deepfake voice datasets with **>95% validation accuracy**.

---

## 📁 Repository Structure

```
Mini_project-III/
├── app.py                     # Streamlit web application
├── model.py                   # CNN architecture definition
├── preprocessing.py           # Audio loading, normalization & Mel-spectrogram extraction
├── train.py                   # Model training pipeline & checkpoint management
├── debug_prediction.py        # CLI test and prediction script
├── requirements.txt           # Python dependencies
├── run.bat                    # One-click environment setup & run script
├── utils/
│   ├── explainability.py      # Grad-CAM implementation & heatmap overlay
│   └── setup_kaggle_dataset.py# Automated dataset validator & extractor
└── models/
    ├── deepfake_audio_detector.h5 # Trained Keras CNN model
    └── training_history.png       # Accuracy & loss curves
```

---

## 🚀 Quick Start

### 1. Automated Setup (Windows)
Double-click `run.bat` or run:
```cmd
run.bat
```

### 2. Manual Setup
```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Streamlit app
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🧠 Model Architecture & Methodology

1. **Audio Preprocessing**:
   - Resampled to 16,000 Hz Mono.
   - Silence trimming & amplitude normalization.
   - 128-band Mel-Spectrogram extraction (2048 FFT, 512 hop length) scaled to $[0, 1]$.
2. **CNN Architecture**:
   - 4 Convolutional Blocks ($32 \to 64 \to 128 \to 128$ filters) with ReLU activations and Dropout regularization.
   - Dense representation layer with 512 units.
   - Softmax output layer for Real vs. AI Deepfake classification.
3. **Explainability**:
   - Gradient-weighted Class Activation Mapping (Grad-CAM) extracts activation gradients from the final convolutional layer.
   - Blends with the input spectrogram using a colormap overlay to identify suspicious frequency bands.