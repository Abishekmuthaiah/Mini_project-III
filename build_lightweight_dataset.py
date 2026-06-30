import os
import numpy as np
import librosa
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')

real_dir = "dataset/real"
fake_dir = "dataset/fake"
os.makedirs(real_dir, exist_ok=True)
os.makedirs(fake_dir, exist_ok=True)

print("Fetching authentic Human Speech arrays from Librosa Core Library...")

def add_noise(data, noise_factor):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    return augmented_data

try:
    humans = ['libri1', 'libri2', 'libri3']
    for i, file_id in enumerate(humans):
        y, sr = librosa.load(librosa.example(file_id), sr=16000)
        
        # Take 10 second chunks to speed things up
        chunk_len = sr * 10
        y = y[:chunk_len] if len(y) > chunk_len else y
        
        # Generate 40 overlapping heavily augmented variations
        for k in range(40):
            idx = i * 40 + k
            
            y_aug = np.copy(y)
            
            # Substantial pitch shift to generalize vocal range (male/female/child)
            shift_steps = np.random.uniform(-5, 5)
            y_aug = librosa.effects.pitch_shift(y=y_aug, sr=sr, n_steps=shift_steps)
            
            # Massive background noise to simulate terrible built-in notebook microphones
            y_aug = add_noise(y_aug, np.random.uniform(0.005, 0.05))
            
            sf.write(os.path.join(real_dir, f"human_{idx}.wav"), y_aug, sr)
            
    print("Generated 120 authentic human vocal spectrogram references covering low-quality microphone environments.")

except Exception as e:
    print(f"Failed to load Librosa speech: {e}")

# Generate Synthetic/Fake samples (Robotic sine combinations & White Noise)
print("Generating synthetic AI patterns...")
for i in range(120):
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # highly unnatural composite signals simulating raw AI synthesis glitch artifacts
    freq1 = np.random.uniform(5500, 6500)
    freq2 = np.random.uniform(7000, 8000)
    signal = 0.5 * np.sin(2 * np.pi * freq1 * t * (1.1 + np.sin(2*np.pi*t))) 
    signal += 0.5 * np.sin(2 * np.pi * freq2 * t)
    
    # Adding synthetic digital clipping
    signal = np.clip(signal, -0.2, 0.2) 
    
    sf.write(os.path.join(fake_dir, f"ai_{i}.wav"), signal, sr)
    
print("Generated Fake audio signatures.")
print("Dataset build complete!")
