import numpy as np
import cv2
import streamlit as st
from PIL import Image
import os
import requests
import time

# Page configuration
st.set_page_config(
    page_title="AI Image Colorizer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom CSS
def load_css():
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Hugging Face direct download URLs
URLS = {
    "prototxt": "https://huggingface.co/bsp1335/colorization-models/resolve/main/models/models_colorization_deploy_v2.prototxt",
    "model": "https://huggingface.co/bsp1335/colorization-models/resolve/main/models/colorization_release_v2.caffemodel",
    "points": "https://huggingface.co/bsp1335/colorization-models/resolve/main/models/pts_in_hull.npy"
}

# Local paths
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

@st.cache_data
def download_file(url, save_path):
    """Download file with caching to avoid repeated downloads"""
    if not os.path.exists(save_path):
        with st.spinner(f"Downloading {os.path.basename(save_path)}..."):
            with requests.get(url, stream=True) as r:
                with open(save_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
    return save_path

# Download required model files
@st.cache_data
def ensure_models_downloaded():
    """Ensure all model files are downloaded"""
    for name, url in URLS.items():
        filename = os.path.basename(url)
        filepath = os.path.join(MODEL_DIR, filename)
        download_file(url, filepath)
    return True

@st.cache_data
def colorizer(img):
    """Colorize a grayscale image using deep learning model"""
    # Convert to grayscale if not already
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    prototxt = os.path.join(MODEL_DIR, "models_colorization_deploy_v2.prototxt")
    model = os.path.join(MODEL_DIR, "colorization_release_v2.caffemodel")
    points = os.path.join(MODEL_DIR, "pts_in_hull.npy")

    # Load the model
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    pts = np.load(points)

    # Add cluster centers as 1x1 convolution kernel
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    # Preprocess the image
    scaled = img.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # Run the model
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    return colorized

# Main UI
def main():
    # Header
    st.title("🎨 AI Image Colorizer")
    st.markdown(
        """
        <div style='text-align: center; margin-bottom: 2rem;'>
            Transform your black & white photos into vibrant, colorized images using advanced AI technology.
            Simply upload an image and watch the magic happen!
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Ensure models are downloaded
    ensure_models_downloaded()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "png", "jpeg"],
        help="Upload a black & white or color image. The AI will automatically colorize it."
    )

    if uploaded_file is not None:
        # Load and display the original image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Create columns for side-by-side display
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.image(image, caption="📷 Original Image", use_column_width=True)
            
        with col2:
            # Show processing message with spinner
            with st.spinner("🎨 AI is working its magic..."):
                # Add a small delay for better UX
                time.sleep(0.5)
                
                # Process the image
                try:
                    colorized = colorizer(img_array)
                    st.image(colorized, caption="✨ Colorized Result", use_column_width=True)
                    
                    # Success message
                    st.success("🎉 Colorization complete! Your image has been transformed.")
                    
                    # Optional: Add download button for the result
                    colorized_pil = Image.fromarray(colorized)
                    
                except Exception as e:
                    st.error(f"❌ An error occurred during processing: {str(e)}")
                    st.info("💡 Try uploading a different image or check if the file is corrupted.")
    
    else:
        # Show example or instructions when no file is uploaded
        st.info("👆 Upload an image above to get started!")
        
        # Add some example text
        st.markdown(
            """
            ### How it works:
            1. **Upload** a black & white or color image
            2. **Wait** for the AI to process your image
            3. **Download** your beautifully colorized result
            
            ### Tips for best results:
            - Use high-quality images for better colorization
            - Black & white photos work exceptionally well
            - The AI has been trained on diverse image types
            """
        )

if __name__ == "__main__":
    main()
