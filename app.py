import numpy as np
import cv2
import streamlit as st
from PIL import Image
import os
import requests

# Hugging Face direct download URLs (replace 'bsp1335' with your Hugging Face username if different)
URLS = {
    "prototxt": "https://huggingface.co/bsp1335/colorization-models/resolve/main/models/models_colorization_deploy_v2.prototxt",
    "model": "https://huggingface.co/bsp1335/colorization-models/resolve/main/models/colorization_release_v2.caffemodel",
    "points": "https://huggingface.co/bsp1335/colorization-models/resolve/main/models/pts_in_hull.npy"
}

# Local paths
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def download_file(url, save_path):
    if not os.path.exists(save_path):
        with requests.get(url, stream=True) as r:
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

# Download required model files
for name, url in URLS.items():
    filename = os.path.basename(url)
    filepath = os.path.join(MODEL_DIR, filename)
    download_file(url, filepath)

def colorizer(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    prototxt = os.path.join(MODEL_DIR, "models_colorization_deploy_v2.prototxt")
    model = os.path.join(MODEL_DIR, "colorization_release_v2.caffemodel")
    points = os.path.join(MODEL_DIR, "pts_in_hull.npy")

    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    pts = np.load(points)

    # Add cluster centers as 1x1 convolution kernel
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    scaled = img.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    return colorized

# Streamlit UI
st.title("Colorizing Black & White Images")
st.write("Upload a black and white image to see it colorized")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)

    with col2:
        st.text("Colorizing...")
        colorized = colorizer(img)
        st.image(colorized, caption="Colorized Image", use_column_width=True)
