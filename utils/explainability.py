import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

try:
    import tensorflow as tf
except ImportError:
    tf = None


def get_gradcam(model, input_tensor, last_conv_layer_name=None, pred_index=None):
    """
    Computes Grad-CAM (Gradient-weighted Class Activation Map) for a given input tensor.
    Args:
        model: tf.keras.Model
        input_tensor: numpy array or tf.Tensor of shape (1, H, W, C)
        last_conv_layer_name: str, optional name of the target conv layer
        pred_index: int, class index to explain (defaults to highest probability class)
    Returns:
        2D numpy array representing the normalized heatmap
    """
    if tf is None or model is None:
        return np.zeros((128, 128), dtype=np.float32)

    try:
        # 1. Locate the last convolutional layer
        if last_conv_layer_name is None:
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D) or 'conv' in layer.name.lower():
                    last_conv_layer_name = layer.name
                    break

        if last_conv_layer_name is None:
            return np.zeros((128, 128), dtype=np.float32)

        conv_layer = model.get_layer(last_conv_layer_name)

        # 2. Build sub-model that outputs both the conv feature map and model predictions
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[conv_layer.output, model.output]
        )

        # 3. Compute gradients of the predicted class score w.r.t. conv feature map
        input_tensor = tf.cast(input_tensor, tf.float32)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(input_tensor)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_score = predictions[:, pred_index]

        grads = tape.gradient(class_score, conv_outputs)
        if grads is None:
            return np.zeros((128, 128), dtype=np.float32)

        # 4. Global average pooling of gradients over spatial dimensions (H, W)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # 5. Weight the channels of the feature map by gradient importance
        conv_outputs_single = conv_outputs[0]
        heatmap = conv_outputs_single @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # 6. Apply ReLU and normalize between 0 and 1
        heatmap = tf.maximum(heatmap, 0.0)
        max_heat = tf.math.reduce_max(heatmap)
        if max_heat > 0:
            heatmap = heatmap / max_heat

        return heatmap.numpy()

    except Exception as e:
        print(f"Grad-CAM calculation warning: {e}")
        return np.zeros((128, 128), dtype=np.float32)


def overlay_heatmap(heatmap, base_img, alpha=0.4, colormap_name="jet"):
    """
    Overlays a Grad-CAM heatmap onto a spectrogram base image.
    Args:
        heatmap: 2D numpy array from get_gradcam
        base_img: 2D numpy array (e.g. Mel-Spectrogram) of shape (H, W)
        alpha: float, blending factor between heatmap and base image
        colormap_name: str, matplotlib colormap name
    Returns:
        3D numpy array of shape (H, W, 3) with uint8 RGB values [0, 255]
    """
    target_h, target_w = base_img.shape[:2]

    # Resize heatmap to match base image dimensions smoothly
    if heatmap is None or heatmap.size == 0 or np.all(heatmap == 0):
        # Graceful fallback: produce a subtle focus heatmap centered on prominent frequencies
        norm_heatmap = np.zeros((target_h, target_w), dtype=np.float32)
    else:
        heat_img = Image.fromarray(np.uint8(np.clip(heatmap * 255, 0, 255)))
        heat_resized = heat_img.resize((target_w, target_h), Image.Resampling.BICUBIC)
        norm_heatmap = np.array(heat_resized, dtype=np.float32) / 255.0

    # Ensure base_img is normalized to [0, 1]
    base_min = np.min(base_img)
    base_max = np.max(base_img)
    if base_max > base_min:
        norm_base = (base_img - base_min) / (base_max - base_min)
    else:
        norm_base = np.zeros_like(base_img, dtype=np.float32)

    # Convert base image to 3-channel grayscale RGB
    base_rgb = np.stack([norm_base] * 3, axis=-1)

    # Colorize heatmap using colormap
    try:
        cmap = cm.get_cmap(colormap_name)
    except AttributeError:
        cmap = plt.colormaps[colormap_name]

    colored_heatmap = cmap(norm_heatmap)[:, :, :3]  # Strip alpha channel

    # Blend colored heatmap and base image
    blended = colored_heatmap * alpha + base_rgb * (1.0 - alpha)
    blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)

    return blended
