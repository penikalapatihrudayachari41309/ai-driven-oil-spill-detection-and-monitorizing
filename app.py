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
    os.getenv("MODEL_PATH"),  # allow user to set environment variable
    os.path.join(os.path.dirname(__file__), MODEL_FILENAME),  # same folder as app.py
    os.path.join(os.path.dirname(__file__), "models", MODEL_FILENAME),  # models/ subfolder
    os.path.join("/app", MODEL_FILENAME),  # some hosts use /app
]

# Environment variable used to optionally download a model at startup
MODEL_DOWNLOAD_URL = os.getenv("MODEL_DOWNLOAD_URL")

# --- Helper Functions --- #
@st.cache_resource
def load_model_from_path(model_path: str):
    """Load a keras model from a filesystem path. Returns model or None."""
    if not model_path:
        return None
    try:
        # set mixed precision if desired
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        except Exception:
            pass

        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.exception(f"Error loading model from {model_path}: {e}")
        return None


def download_model_if_needed(download_url: str, save_dir: str, filename: str):
    """Download the model from a URL into save_dir/filename if not already present."""
    if not download_url:
        return None

    os.makedirs(save_dir, exist_ok=True)
    dest_path = os.path.join(save_dir, filename)

    if os.path.exists(dest_path):
        return dest_path

    try:
        with requests.get(download_url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return dest_path
    except Exception as e:
        st.warning(f"Failed to download model from {download_url}: {e}")
        return None


def find_existing_model_path():
    """Return the first existing path from DEFAULT_MODEL_PATHS or None."""
    for p in DEFAULT_MODEL_PATHS:
        if p and os.path.exists(p):
            return p
    return None


def preprocess_image_for_prediction(image):
    img_np = np.array(image.convert('L'))
    img_resized = PIL.Image.fromarray(img_np).resize(TARGET_SIZE, PIL.Image.LANCZOS)
    img_normalized = img_resized / 255.0
    img_processed = np.expand_dims(img_normalized, axis=(0, -1))
    return img_processed


def postprocess_mask_prediction(prediction_output, original_size):
    mask_np = prediction_output.squeeze()
    mask_binary = (mask_np > 0.5).astype(np.float32)
    mask_pil = PIL.Image.fromarray((mask_binary * 255).astype(np.uint8))
    mask_resized_to_original = mask_pil.resize(original_size, PIL.Image.NEAREST)
    return np.array(mask_resized_to_original)


def overlay_mask_on_image(original_image_rgb, predicted_mask, color=(0, 255, 0), alpha=0.5):
    if predicted_mask.ndim == 3 and predicted_mask.shape[-1] == 1:
        predicted_mask = predicted_mask.squeeze()

    h, w = original_image_rgb.shape[:2]
    colored_overlay = np.zeros((h, w, 3), dtype=np.uint8)

    mask_bool = predicted_mask > 0

    for c in range(3):
        colored_overlay[mask_bool, c] = color[c]

    blended_image = (
        original_image_rgb * (1 - alpha) + colored_overlay * alpha
    )
    blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)
    return blended_image


# --- Streamlit UI --- #
def app_main():
    st.title("Oil Spill Detection Application")
    st.write("Upload a SAR image to detect oil spills.")

    model = None

    # Try downloading model if URL provided
    if MODEL_DOWNLOAD_URL:
        st.info(f"Attempting to download model from: {MODEL_DOWNLOAD_URL}")
        downloaded = download_model_if_needed(
            MODEL_DOWNLOAD_URL,
            os.path.join(os.path.dirname(__file__), 'models'),
            MODEL_FILENAME
        )
        if downloaded:
            st.success(f"Model downloaded to: {downloaded}")
            model = load_model_from_path(downloaded)
        else:
            st.warning("Model download failed.")

    # Try loading existing model
    if model is None:
        existing_model_path = find_existing_model_path()
        if existing_model_path:
            st.info(f"Loading model from: {existing_model_path}")
            model = load_model_from_path(existing_model_path)
        else:
            st.warning(
                "No model found. Upload a model below or set MODEL_PATH / MODEL_DOWNLOAD_URL."
            )

    # Upload model manually
    st.subheader("Model")
    uploaded_model_file = st.file_uploader(
        "Upload .keras or .h5 model (optional)",
        type=["keras", "h5", "zip"]
    )

    if uploaded_model_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_model_file.name)[1]) as tfp:
            tfp.write(uploaded_model_file.getvalue())
            tmp_model_path = tfp.name

        st.info(f"Loading uploaded model: {uploaded_model_file.name}")
        model = load_model_from_path(tmp_model_path)

    uploaded_file = st.file_uploader("Choose an image..", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        original_image = PIL.Image.open(uploaded_file)
        st.subheader("Original Image")
        st.image(original_image, caption='Uploaded Image', use_column_width=True)

        original_size = original_image.size

        if model is not None:
            with st.spinner("Processing image..."):
                try:
                    processed_image = preprocess_image_for_prediction(original_image)
                    prediction = model.predict(processed_image)
                    predicted_mask_output = postprocess_mask_prediction(prediction, original_size)

                    if np.any(predicted_mask_output > 0):
                        st.success("Oil Spill Detected!")
                    else:
                        st.info("No Oil Spill Detected.")

                    original_image_rgb = np.array(original_image.convert('RGB'))
                    overlaid_img = overlay_mask_on_image(original_image_rgb, predicted_mask_output)

                    st.subheader("Detection Results")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.image(predicted_mask_output, caption='Predicted Mask', use_column_width=True)
                    with col2:
                        st.image(overlaid_img, caption='Overlaid Prediction', use_column_width=True)

                    st.subheader("Download Results")
                    col_dl1, col_dl2 = st.columns(2)

                    with col_dl1:
                        mask_pil = PIL.Image.fromarray(predicted_mask_output.astype(np.uint8))
                        buf = BytesIO()
                        mask_pil.save(buf, format='PNG')
                        st.download_button(
                            "Download Predicted Mask",
                            buf.getvalue(),
                            "predicted_mask.png",
                            "image/png"
                        )

                    with col_dl2:
                        overlaid_pil = PIL.Image.fromarray(overlaid_img)
                        buf = BytesIO()
                        overlaid_pil.save(buf, format='PNG')
                        st.download_button(
                            "Download Overlaid Image",
                            buf.getvalue(),
                            "overlaid_prediction.png",
                            "image/png"
                        )

                except Exception as e:
                    st.exception(e)
        else:
            st.error("Model not loaded. Upload a model or fix MODEL_PATH.")

if __name__ == '__main__':
    app_main()
