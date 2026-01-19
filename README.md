# AI-Driven Oil Spill Detection and Monitoring Application

A deep learning-based Streamlit application for detecting oil spills in SAR (Synthetic Aperture Radar) satellite imagery using a U-Net segmentation model.

## 🌐 Live Demo

Access the deployed application at: **https://ai-oil-spill-detection-and-monitoring-pha.streamlit.app/**

## 🚀 Deployment Options

### Option 1: Streamlit Cloud (Recommended for Permanent Hosting)

1. Push this repository to GitHub
2. Go to https://share.streamlit.io/
3. Connect your GitHub repository
4. The app will automatically deploy

**Required files:**
- `app.py` - Main application code
- `requirements.txt` - Python dependencies
- `unet_oil_spill_segmentation_model_128x128.keras` - Trained model
- `.streamlit/config.toml` - Streamlit configuration

### Option 2: Local Testing with ngrok

```bash
# Install dependencies
pip install -r requirements.txt

# Install ngrok
pip install pyngrok

# Run with ngrok for public URL
python run_with_ngrok.py
```

### Option 3: Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📁 Project Structure

```
ai-driven-oil-spill-detection-and-monitorizing/
├── app.py                                    # Main Streamlit application
├── run_with_ngrok.py                         # Script for ngrok deployment
├── requirements.txt                          # Python dependencies
├── unet_oil_spill_segmentation_model_128x128.keras  # Trained U-Net model
├── .streamlit/
│   └── config.toml                           # Streamlit configuration
├── README.md                                 # This file
└── LICENSE                                   # License
```

## 🛠️ Features

- **Image Upload**: Upload SAR satellite images (PNG, JPG, JPEG)
- **Real-time Detection**: Automatic oil spill segmentation using U-Net
- **Visual Results**: Display original image, predicted mask, and overlaid visualization
- **Classification**: Displays "Oil Spill Detected" or "No Oil Spill Detected"
- **Download Results**: Download predicted mask and overlaid images
- **Cross-platform**: Works on local machines and Streamlit Cloud

## 📊 Model Information

- **Architecture**: U-Net with reduced filters (32-64-128-256)
- **Input Size**: 128x128 grayscale images
- **Training**: Mixed precision training enabled
- **Dataset**: Deep SAR (SOS) Dataset
- **Performance**: ~91.68% binary accuracy on test set

## 📦 Dependencies

```
tensorflow==2.19.0
numpy==2.0.2
Pillow==11.3.0
streamlit
albumentations==1.3.1
scikit-image==0.25.2
opencv-python==4.12.0.88
pyngrok
matplotlib
```

## 🔧 Troubleshooting

### Model File Not Found

If you see the error:
```
Error loading model: File not found: filepath=/content/drive/...
```

Ensure the model file `unet_oil_spill_segmentation_model_128x128.keras` is in the same directory as `app.py`.

### Memory Issues

If you encounter GPU memory errors:
- Reduce batch size in the model configuration
- Use CPU inference by setting `os.environ['CUDA_VISIBLE_DEVICES'] = '-1'`

## 📝 License

This project is licensed under the MIT License.

## 👨‍💻 Author

Developed for environmental monitoring and oil spill detection using satellite imagery analysis.

