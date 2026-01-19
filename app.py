import streamlit as st
import tensorflow as tf
import numpy as np
import PIL.Image
import os
from io import BytesIO
import tempfile
import requests

# --- Configuration --- #
TARGET_SIZE = (128, 128)
MODEL_FILENAME = f'unet_oil_spill_segmentation_model_{TARGET_SIZE[0]}x{TARGET_SIZE[1]}.keras'

# Candidate default paths (check in this order)
DEFAULT_MODEL_PATHS = [
    os.getenv("MODEL_PATH"),                           # allow user to set environment variable
    os.path.join(os.path.dirname(__file__), MODEL_FILENAME),  # same folder as app.py
    os.path.join(os.path.dirname(__file__), "models", MODEL_FILENAME),  # models/ subfolder
    os.path.join("/app", MODEL_FILENAME),              # some hosts use /app
]

# Hardcoded model download URL (Drive direct download). This will be used unless you set the MODEL_DOWNLOAD_URL environment variable.
MODEL_DOWNLOAD_URL = os.getenv("MODEL_DOWNLOAD_URL", "https://drive.google.com/uc?export=download&id=1HhwuiRyNsDk8rw48uHfDxYiKe0tdbnlA")

# --- Helper Functions --- #
@st.cache_resource
def load_model_from_path(model_path: str):
    """Load a keras model from a filesystem path. Returns model or None."""
    if not model_path:
        return None
    try:
        # set mixed precision if desired (wrapped to avoid errors on unsupported TF versions)
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        except Exception:
            # not fatal; continue without mixed precision
            pass
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        # Surface the exception message to UI (streamlit will also print to logs)
        st.exception(f"Error loading model from {model_path}: {e}")
        return None


def download_model_if_needed(download_url: str, save_dir: str, filename: str):
    """Download the model from a URL into save_dir/filename if not already present. Returns path or None."""
    if not download_url or "drive.google.com/uc?export=download" not in download_url:
        # still attempt for other URLs; keep the check minimal
        pass
    os.makedirs(save_dir, exist_ok=True)
    dest_path = os.path.join(save_dir, filename)
    if os.path.exists(dest_path):
        return dest_path
    try:
        with requests.get(download_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            chunk_size = 8192
            with open(dest_path, 'wb') as f:
                # show a simple progress in logs; Streamlit progress requires main thread
                downloaded = 0
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
        return dest_path
    except Exception as e:
        st.warning(f"Failed to download model from {download_url}: {e}")
        return None


def find_existing_model_path():
    """Return the first existing path from DEFAULT_MODEL_PATHS or None."""
    for p in DEFAULT_MODEL_PATHS:
        if not p:
            continue
        if os.path.exists(p):
            return p
    return None


def preprocess_image_for_prediction(image):
    img_np = np.array(image.convert('L'))
    img_resized = PIL.Image.fromarray(img_np).resize(TARGET_SIZE, PIL.Image.LANCZOS)
    img_np_resized = np.array(img_resized)
    img_normalized = img_np_resized / 255.0
    img_processed = np.expand_dims(img_normalized, axis=(0, -1))
    return img_processed


def postprocess_mask_prediction(prediction_output, original_size):
    mask_np = prediction_output.squeeze()
    mask_binary = (mask_np > 0.5).astype(np.float32)
    mask_pil = PIL.Image.fromarray((mask_binary * 255).astype(np.uint8))
    # original_size is (width, height) from PIL Image.size
    mask_resized_to_original = mask_pil.resize(original_size, PIL.Image.NEAREST)
    return np.array(mask_resized_to_original)  # Return 0-255 uint8 for consistent image saving/display


def overlay_mask_on_image(original_image_rgb, predicted_mask, color=(0, 255, 0), alpha=0.5):
    if predicted_mask.ndim == 3 and predicted_mask.shape[-1] == 1:
        predicted_mask = predicted_mask.squeeze(axis=-1)

    h, w = original_image_rgb.shape[:2]
    colored_overlay = np.zeros((h, w, 3), dtype=np.uint8)

    # predicted_mask should be 0..255, convert to boolean
    mask_bool = predicted_mask > 0

    for c in range(3):
        colored_overlay[mask_bool, c] = color[c]

    blended_image = (1 - alpha) * original_image_rgb.astype(float) + alpha * colored_overlay.astype(float)
    blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)
    return blended_image


# --- Streamlit UI --- #
def app_main():
    st.title("Oil Spill Detection Application")
    st.write("Upload a SAR image to detect oil spills.")

    # If MODEL_DOWNLOAD_URL is provided (or hardcoded), attempt to download into ./models/
    model = None
    if MODEL_DOWNLOAD_URL and "drive.google.com/uc?export=download" in MODEL_DOWNLOAD_URL:
        st.info("MODEL_DOWNLOAD_URL is configured. Attempting to download model from Drive link...")
        downloaded = download_model_if_needed(MODEL_DOWNLOAD_URL, os.path.join(os.path.dirname(__file__), 'models'), MODEL_FILENAME)
        if downloaded:
            st.success(f"Model downloaded to: {downloaded}")
            model = load_model_from_path(downloaded)
        else:
            st.warning("Model download failed or the file was not available at the provided URL. Ensure the Drive file is shared publicly and not too large for Drive's browser confirmation.")

    # Try to find an existing model on disk using common locations or an env var
    if model is None:
        existing_model_path = find_existing_model_path()
        if existing_model_path:
            st.info(f"Loading model from: {existing_model_path}")
            model = load_model_from_path(existing_model_path)
            if model is None:
                st.warning("Found a model path but loading failed. You can set MODEL_DOWNLOAD_URL or add the model to the server.")
        else:
            st.warning(
                "No model file found on the server. You can set the MODEL_DOWNLOAD_URL constant in the app or set the MODEL_PATH env var."
            )

    # Image uploader (no model upload UI — model must come from code/server)
    uploaded_file = st.file_uploader("Choose an image..", type=["png", "jpg", "jpeg"]) 

    if uploaded_file is not None:
        original_image = PIL.Image.open(uploaded_file)
        st.subheader("Original Image")
        st.image(original_image, caption='Uploaded Image', use_column_width=True)

        original_size = original_image.size  # (width, height)

        if model is not None:
            with st.spinner("Processing image and detecting spills..."): 
                try:
                    processed_image = preprocess_image_for_prediction(original_image)
                    # Ensure model returns a prediction shape we can work with
                    prediction = model.predict(processed_image)
                    predicted_mask_output = postprocess_mask_prediction(prediction, original_size)

                    # Determine if oil spill is detected
                    is_spill_detected = np.any(predicted_mask_output > 0)
                    if is_spill_detected:
                        st.success("Oil Spill Detected!")
                    else:
                        st.info("No Oil Spill Detected.")

                    # Convert original image to RGB for overlay function
                    original_image_rgb = np.array(original_image.convert('RGB'))
                    overlaid_img = overlay_mask_on_image(original_image_rgb, predicted_mask_output, color=(0, 255, 0), alpha=0.5)

                    st.subheader("Detection Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(predicted_mask_output, caption='Predicted Mask', use_column_width=True, clamp=True)
                    with col2:
                        st.image(overlaid_img, caption='Overlaid Prediction', use_column_width=True)

                    # --- Download Options ---
                    st.subheader("Download Results")
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        # Convert mask to bytes for download
                        mask_pil_for_dl = PIL.Image.fromarray(predicted_mask_output.astype(np.uint8))
                        mask_bytes = BytesIO()
                        mask_pil_for_dl.save(mask_bytes, format='PNG')
                        st.download_button(
                            label="Download Predicted Mask",
                            data=mask_bytes.getvalue(),
                            file_name="predicted_mask.png",
                            mime="image/png"
                        )
                    with col_dl2:
                        # Convert overlaid image to bytes for download
                        overlaid_pil_for_dl = PIL.Image.fromarray(overlaid_img)
                        overlaid_bytes = BytesIO()
                        overlaid_pil_for_dl.save(overlaid_bytes, format='PNG')
                        st.download_button(
                            label="Download Overlaid Image",
                            data=overlaid_bytes.getvalue(),
                            file_name="overlaid_prediction.png",
                            mime="image/png"
                        )
                except Exception as e:
                    st.exception(f"Error during prediction/display: {e}")
        else:
            st.error("Model could not be loaded. Ensure the Drive file is public and the URL is correct, or place the model on the server.")

if __name__ == '__main__':
    app_main()