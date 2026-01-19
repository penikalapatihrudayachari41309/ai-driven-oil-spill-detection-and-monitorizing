import streamlit as st
import tensorflow as tf
import numpy as np
import PIL.Image
import os
from io import BytesIO

# --- Configuration --- #
TARGET_SIZE = (128, 128)
dataset_path = '/content/drive/MyDrive/Deep SAR (SOS) Dataset/'
model_save_path = os.path.join(dataset_path, f'unet_oil_spill_segmentation_model_{TARGET_SIZE[0]}x{TARGET_SIZE[1]}.keras')

# --- Helper Functions --- #
@st.cache_resource
def load_model():
    try:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        model = tf.keras.models.load_model(model_save_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.warning("Please ensure the model path is correct and the model file exists.")
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
    mask_resized_to_original = mask_pil.resize(original_size, PIL.Image.NEAREST)
    return np.array(mask_resized_to_original) # Return 0-255 uint8 for consistent image saving/display

def overlay_mask_on_image(original_image_rgb, predicted_mask, color=(0, 255, 0), alpha=0.5):
    if predicted_mask.ndim == 3 and predicted_mask.shape[-1] == 1:
        predicted_mask = predicted_mask.squeeze(axis=-1)

    h, w = original_image_rgb.shape[:2]
    colored_overlay = np.zeros((h, w, 3), dtype=np.uint8)

    mask_bool = predicted_mask > 0.5

    for c in range(3):
        colored_overlay[mask_bool, c] = color[c]

    blended_image = (1 - alpha) * original_image_rgb.astype(float) + alpha * colored_overlay.astype(float)
    blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)
    return blended_image

# --- Streamlit UI --- #
def app_main():
    st.title("Oil Spill Detection Application")
    st.write("Upload a SAR image to detect oil spills.")

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        original_image = PIL.Image.open(uploaded_file)
        st.subheader("Original Image")
        st.image(original_image, caption='Uploaded Image', use_column_width=True)

        original_size = original_image.size # (width, height)

        model = load_model()

        if model is not None:
            with st.spinner("Processing image and detecting spills..."):
                processed_image = preprocess_image_for_prediction(original_image)
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
        else:
            st.error("Model could not be loaded. Please check the console for errors.")

if __name__ == "__main__":
    app_main()
