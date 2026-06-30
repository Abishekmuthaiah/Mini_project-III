import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import preprocessing
import model

# Labels
# 0: Real
# 1: Fake

def load_dataset(base_dir="dataset", target_shape=(128, 128)):
    """
    Scans the dataset directory, processes all audio files into spectrograms,
    and constructs feature (X) and label (y) arrays.
    """
    X = []
    y = []
    
    classes = {"real": 0, "fake": 1}
    
    for cls_name, cls_label in classes.items():
        cls_dir = os.path.join(base_dir, cls_name)
        if not os.path.exists(cls_dir):
            print(f"Warning: Dataset directory {cls_dir} not found.")
            continue
            
        for filename in os.listdir(cls_dir):
            if filename.endswith(".wav") or filename.endswith(".mp3"):
                filepath = os.path.join(cls_dir, filename)
                try:
                    # 1. Load Audio
                    audio_y, sr = preprocessing.load_audio(filepath)
                    # 2. Extract Spectrogram Face
                    # Assuming default 128 max pad length
                    spec = preprocessing.extract_mel_spectrogram(audio_y, sr)
                    
                    X.append(spec)
                    y.append(cls_label)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Loaded {len(X)} samples total.")
    return X, y

def plot_history(history, save_path="models/training_history.png"):
    """
    Plots training and validation accuracy/loss and saves as image.
    """
    acc = history.history['accuracy']
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history['loss']
    val_loss = history.history.get('val_loss', [])

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    if val_acc:
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    if val_loss:
        plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved training history to {save_path}")

def main():
    print("Loading Dataset...")
    X, y = load_dataset()
    
    if len(X) == 0:
        print("No valid data loaded. Please ensure dataset/real and dataset/fake have audio files.")
        return
        
    print(f"Input shape: {X.shape}, Label shape: {y.shape}")

    # Train / Val Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training split: {len(X_train)} samples")
    print(f"Validation split: {len(X_val)} samples")
    
    cnn_model = model.build_model(input_shape=(X.shape[1], X.shape[2], 1))
    
    # Callbacks
    os.makedirs("models", exist_ok=True)
    checkpoint_path = "models/deepfake_audio_detector.h5"
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    print("Starting training...")
    history = cnn_model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stop]
    )
    
    # Save training graphs
    plot_history(history)
    print("Training finished!")

if __name__ == "__main__":
    main()
