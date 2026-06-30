import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape=(128, 128, 1)):
    """
    Builds and compiles a Convolutional Neural Network for Deepfake Audio Detection.
    Args:
        input_shape (tuple): Dimension of the input spectrogram image (Height, Width, Channels)
    Returns:
        tf.keras.Model: Compiled CNN model
    """
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 4
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Classification Head
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Softmax Output: 2 Classes (Real vs Fake)
    outputs = layers.Dense(2, activation='softmax')(x)

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',  # Target labels will be integers (0/1)
                  metrics=['accuracy'])

    return model

if __name__ == "__main__":
    # Test model build
    m = build_model()
    m.summary()
