import streamlit as st
import tensorflow as tf
import numpy as np
import PIL.Image
import os
import zipfile
from io import BytesIO

# --- Configuration --- #
TARGET_SIZE = (128, 128)

MODEL_ZIP = f"unet_oil_spill_segmentation_model_{TARGET_SIZE[0]}x{TARGET_SIZE[1]}.zip"
MODEL_FILENAME = f"unet_oil_spill_segmentation_model_{TARGET_SIZE[0]}x{TARGET_SIZE[1]}.keras"

BASE_DIR = os.path.dirname(__file__)
EXTRACT_DIR = os.path.join(BASE_DIR, "extracted_model")

# Possible ZIP locations in repo
MODEL_ZIP_PATHS = [
    os.path.join(BASE_DIR, MODEL_ZIP),
    os.path.join(BASE_DIR, "models", MODEL_ZIP),
]

# --- Helper Functions --- #
@st.cache_resource
def extract_and_load_model():
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    zip_path = None
    for p in MODEL_ZIP_PATHS:
        if os.path.exists(p):
            zip_path = p
            break

    if zip_path is None:
        st.error("Model ZIP file not found in repository.")
        return None

    # Extract ZIP only once
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

    # Find .keras file
    for root, _, files in os.walk(EXTRACT_DIR):
        for f in files:
            if f.endswith(".keras"):
                model_path = os.path.join(root, f)
                try:
                    try:
                        tf.keras.mixed_precision.set_global_policy("mixed_float16")
                    except Exception:
                        pass

                    return tf.keras.models.load_model(model_path, compile=False)
                except Exception as e:
                    st.exception(f"Failed to load model: {e}")
                    return None

    st.error("No .keras model found inside ZIP.")
    return None


def preprocess_image_for_prediction(image):
    img = np.array(image.convert("L"))
    img = PIL.Image.fromarray(img).resize(TARGET_SIZE, PIL.Image.LANCZOS)
    img = img / 255.0
    return np.expand_dims(img, axis=(0, -1))


def postprocess_mask_prediction(prediction, original_size):
    mask = (prediction.squeeze() > 0.5).astype(np.uint8) * 255
    mask = PIL.Image.fromarray(mask).resize(original_size, PIL.Image.NEAREST)
    return np.array(mask)


def overlay_mask_on_image(image_rgb, mask, color=(0, 255, 0), alpha=0.5):
    overlay = np.zeros_like(image_rgb)
    mask_bool = mask > 0

    for c in range(3):
        overlay[..., c][mask_bool] = color[c]

    blended = image_rgb * (1 - alpha) + overlay * alpha
    return blended.astype(np.uint8)


# --- Streamlit UI --- #
def app_main():
    st.title("Oil Spill Detection Application")
    st.write("Upload a SAR image to detect oil spills.")

    # ✅ Load model from ZIP in repo
    model = extract_and_load_model()

    if model:
        st.success("Model loaded successfully from ZIP file.")

    uploaded_file = st.file_uploader(
        "Choose an image", type=["png", "jpg", "jpeg"]
    )

    if uploaded_file and model:
        image = PIL.Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Detecting oil spill..."):
            processed = preprocess_image_for_prediction(image)
            prediction = model.predict(processed)
            mask = postprocess_mask_prediction(prediction, image.size)

            if np.any(mask > 0):
                st.success("Oil Spill Detected!")
            else:
                st.info("No Oil Spill Detected.")

            overlay = overlay_mask_on_image(
                np.array(image.convert("RGB")),
                mask
            )

            col1, col2 = st.columns(2)
            with col1:
                st.image(mask, caption="Predicted Mask", use_column_width=True)
            with col2:
                st.image(overlay, caption="Overlay Result", use_column_width=True)

            # Download
            st.subheader("Download Results")
            buf = BytesIO()
            PIL.Image.fromarray(mask).save(buf, format="PNG")
            st.download_button(
                "Download Mask",
                buf.getvalue(),
                "predicted_mask.png",
                "image/png"
            )


if __name__ == "__main__":
    app_main()
