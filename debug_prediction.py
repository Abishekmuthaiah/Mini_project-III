import os
import numpy as np
import tensorflow as tf
import preprocessing

model_path = "models/deepfake_audio_detector.h5"
model = tf.keras.models.load_model(model_path)

def test_file(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
        
    y, sr = preprocessing.load_audio(filepath)
    spec_array = preprocessing.extract_mel_spectrogram(y, sr)
    input_tensor = np.expand_dims(spec_array, axis=0)
    
    prediction = model.predict(input_tensor, verbose=0)[0]
    pred_class = int(np.argmax(prediction))
    
    label = "Real" if pred_class == 0 else "Fake"
    print(f"Testing {os.path.basename(filepath)}:")
    print(f"  Raw Prediction Array: {prediction}")
    print(f"  Predicted Class: {pred_class} ({label})")
    print("-" * 40)

# Test a real and fake file
real_files = [f for f in os.listdir("dataset/real") if f.endswith(".wav")]
fake_files = [f for f in os.listdir("dataset/fake") if f.endswith(".wav")]

if real_files:
    test_file(os.path.join("dataset/real", real_files[0]))
    test_file(os.path.join("dataset/real", real_files[-1]))
if fake_files:
    test_file(os.path.join("dataset/fake", fake_files[0]))
    test_file(os.path.join("dataset/fake", fake_files[-1]))
