import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

def load_audio(filepath, sr=16000):
    """
    Load an audio file, convert to 16kHz mono, normalize amplitude,
    and remove leading/trailing silence.
    """
    # Load audio
    y, sr = librosa.load(filepath, sr=sr, mono=True)
    
    # Trim silence (top_db controls threshold)
    y, _ = librosa.effects.trim(y, top_db=20)
    
    # Normalize amplitude
    if len(y) > 0:
        y = librosa.util.normalize(y)
        
    return y, sr

def extract_mel_spectrogram(y, sr=16000, n_mels=128, max_pad_len=128):
    """
    Generate Mel-Spectrogram and format it for the CNN.
    Returns a numpy array of shape (n_mels, max_pad_len, 1).
    """
    # Compute Mel-spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=2048, hop_length=512)
    
    # Convert to log scale (decibels)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    
    # Padding or truncating to ensure fixed dimensions
    if log_spectrogram.shape[1] > max_pad_len:
        log_spectrogram = log_spectrogram[:, :max_pad_len]
    else:
        pad_width = max_pad_len - log_spectrogram.shape[1]
        log_spectrogram = np.pad(log_spectrogram, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
    # Min-max scaling between 0 and 1 for CNN
    log_spectrogram = (log_spectrogram - np.min(log_spectrogram)) / (np.max(log_spectrogram) - np.min(log_spectrogram) + 1e-6)
        
    # Add channel dimension (grayscale: 1 channel)
    log_spectrogram = np.expand_dims(log_spectrogram, axis=-1)
    
    return log_spectrogram

def plot_spectrogram(y, sr=16000, title="Mel Spectrogram"):
    """
    Generate a matplotlib Plot for Streamlit UI display.
    """
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(log_spectrogram, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax, cmap='magma')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title(title)
    plt.tight_layout()
    
    return fig
