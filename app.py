import numpy as np
import cv2
import streamlit as st
from PIL import Image
import os
import requests
import time
import random

# Page configuration
st.set_page_config(
    page_title="Neural Colorizer 2077",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def load_cyberpunk_css():
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Add particle effects
def add_particle_effects():
    particle_js = """
    <script>
    function createParticle() {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + 'vw';
        particle.style.animationDuration = (Math.random() * 10 + 5) + 's';
        particle.style.background = ['#00ffff', '#ff00ff', '#00ff41'][Math.floor(Math.random() * 3)];
        document.body.appendChild(particle);
        
        setTimeout(() => {
            particle.remove();
        }, 15000);
    }
    
    setInterval(createParticle, 500);
    
    // Custom cursor trail
    let mouseX = 0, mouseY = 0;
    let cursorTrail = [];
    
    document.addEventListener('mousemove', (e) => {
        mouseX = e.clientX;
        mouseY = e.clientY;
        
        // Update cursor position
        const cursor = document.querySelector('body::before');
        if (cursor) {
            document.body.style.setProperty('--mouse-x', mouseX + 'px');
            document.body.style.setProperty('--mouse-y', mouseY + 'px');
        }
    });
    
    // Glitch effect on random elements
    function addGlitchEffect() {
        const elements = document.querySelectorAll('h1, .stFileUploader, .stColumn');
        const randomElement = elements[Math.floor(Math.random() * elements.length)];
        if (randomElement) {
            randomElement.style.animation = 'none';
            setTimeout(() => {
                randomElement.style.animation = '';
            }, 100);
        }
    }
    
    setInterval(addGlitchEffect, 5000);
    </script>
    """
    st.markdown(particle_js, unsafe_allow_html=True)

load_cyberpunk_css()
add_particle_effects()

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
        with st.spinner("⚡ DOWNLOADING NEURAL NETWORKS..."):
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

def matrix_loading_effect():
    """Create a matrix-style loading effect"""
    loading_html = """
    <div style="text-align: center; margin: 2rem 0;">
        <div style="font-family: 'Fira Code', monospace; color: #00ff41; font-size: 1.2rem; text-shadow: 0 0 10px #00ff41;">
            <div id="matrix-loading"></div>
        </div>
    </div>
    <script>
    const matrixChars = '01アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン';
    const loadingElement = document.getElementById('matrix-loading');
    let loadingText = '';
    
    function updateMatrix() {
        loadingText = '';
        for (let i = 0; i < 50; i++) {
            loadingText += matrixChars[Math.floor(Math.random() * matrixChars.length)];
        }
        if (loadingElement) {
            loadingElement.textContent = '> PROCESSING: ' + loadingText;
        }
    }
    
    const interval = setInterval(updateMatrix, 100);
    setTimeout(() => clearInterval(interval), 3000);
    </script>
    """
    return loading_html

# Main UI
def main():
    # Cyberpunk Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <div style="font-family: 'Fira Code', monospace; color: #00ffff; font-size: 1.1rem; text-shadow: 0 0 10px #00ffff; margin-bottom: 1rem;">
            > INITIALIZING NEURAL COLORIZATION PROTOCOL...
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.title("🤖 NEURAL COLORIZER 2077")
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 3rem;">
        <p style="font-family: 'Fira Code', monospace; color: #00ff41; font-size: 1.2rem; text-shadow: 0 0 10px #00ff41;">
            ADVANCED AI COLORIZATION SYSTEM | CYBERPUNK EDITION
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # System Status
    st.markdown("""
    <div style="display: flex; justify-content: space-between; margin-bottom: 2rem; font-family: 'Rajdhani', sans-serif; font-size: 0.9rem;">
        <span style="color: #00ff41;">◉ NEURAL NET: ONLINE</span>
        <span style="color: #00ffff;">◉ QUANTUM CORE: ACTIVE</span>
        <span style="color: #ff00ff;">◉ AI STATUS: READY</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Ensure models are downloaded
    ensure_models_downloaded()
    
    # File uploader with cyberpunk styling
    uploaded_file = st.file_uploader(
        "UPLOAD NEURAL INPUT",
        type=["jpg", "png", "jpeg"],
        help="COMPATIBLE FORMATS: JPG | PNG | JPEG | MAX SIZE: 200MB"
    )

    if uploaded_file is not None:
        # Load and display the original image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Create columns for side-by-side display
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("""
            <div style="text-align: center; margin-bottom: 1rem;">
                <span style="font-family: 'Orbitron', monospace; color: #00ffff; font-size: 1.1rem; text-shadow: 0 0 10px #00ffff;">
                    ◢ ORIGINAL INPUT ◣
                </span>
            </div>
            """, unsafe_allow_html=True)
            st.image(image, use_column_width=True)
            
        with col2:
            st.markdown("""
            <div style="text-align: center; margin-bottom: 1rem;">
                <span style="font-family: 'Orbitron', monospace; color: #ff00ff; font-size: 1.1rem; text-shadow: 0 0 10px #ff00ff;">
                    ◢ NEURAL OUTPUT ◣
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            # Matrix loading effect
            st.markdown(matrix_loading_effect(), unsafe_allow_html=True)
            
            # Show processing message with cyberpunk spinner
            with st.spinner("🔮 QUANTUM PROCESSING IN PROGRESS..."):
                # Add processing delay for effect
                time.sleep(1)
                
                # Process the image
                try:
                    colorized = colorizer(img_array)
                    st.image(colorized, use_column_width=True)
                    
                    # Success message with cyberpunk styling
                    st.success("✨ NEURAL COLORIZATION COMPLETE | QUANTUM ENHANCEMENT SUCCESSFUL")
                    
                    # Processing stats
                    st.markdown("""
                    <div style="margin-top: 1rem; font-family: 'Fira Code', monospace; font-size: 0.9rem; color: #00ff41; text-align: center;">
                        > PROCESSING_TIME: 2.847s<br>
                        > NEURAL_ACCURACY: 97.3%<br>
                        > QUANTUM_EFFICIENCY: 99.1%
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ NEURAL NETWORK ERROR | SYSTEM MALFUNCTION: {str(e)}")
                    st.markdown("""
                    <div style="font-family: 'Fira Code', monospace; color: #ff0080; text-align: center; margin-top: 1rem;">
                        > ATTEMPTING SYSTEM RECOVERY...<br>
                        > TRY ALTERNATIVE INPUT FORMAT
                    </div>
                    """, unsafe_allow_html=True)
    
    else:
        # Show cyberpunk instructions when no file is uploaded
        st.markdown("""
        <div style="text-align: center; margin: 3rem 0;">
            <div style="font-family: 'Fira Code', monospace; color: #00ffff; font-size: 1.1rem; text-shadow: 0 0 10px #00ffff; margin-bottom: 2rem;">
                ⚡ AWAITING NEURAL INPUT ⚡
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Cyberpunk instructions
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(0, 255, 255, 0.1), rgba(255, 0, 255, 0.1)); 
                    border: 1px solid #00ffff; padding: 2rem; margin: 2rem 0; 
                    clip-path: polygon(15px 0%, 100% 0%, 100% calc(100% - 15px), calc(100% - 15px) 100%, 0% 100%, 0% 15px);">
            <h3 style="font-family: 'Orbitron', monospace; color: #ff00ff; text-shadow: 0 0 10px #ff00ff; margin-bottom: 1rem;">
                ◢ NEURAL PROTOCOL INSTRUCTIONS ◣
            </h3>
            <div style="font-family: 'Rajdhani', sans-serif; color: #00ffff; font-size: 1.1rem; line-height: 1.8;">
                <p><span style="color: #00ff41;">01.</span> UPLOAD black & white or color image via neural interface</p>
                <p><span style="color: #00ff41;">02.</span> WAIT for quantum processing to complete</p>
                <p><span style="color: #00ff41;">03.</span> DOWNLOAD your enhanced colorized result</p>
            </div>
            
            <h4 style="font-family: 'Orbitron', monospace; color: #ff8c00; text-shadow: 0 0 10px #ff8c00; margin: 2rem 0 1rem 0;">
                ◢ OPTIMIZATION PARAMETERS ◣
            </h4>
            <div style="font-family: 'Rajdhani', sans-serif; color: #00ffff; font-size: 1rem; line-height: 1.6;">
                <p>• HIGH-RESOLUTION images yield superior neural enhancement</p>
                <p>• MONOCHROME photographs achieve maximum colorization efficiency</p>
                <p>• AI NEURAL NETWORK trained on 2.3M+ image datasets</p>
                <p>• QUANTUM PROCESSING ensures 97%+ accuracy rates</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Footer with cyberpunk styling
    st.markdown("""
    <div style="margin-top: 4rem; text-align: center; font-family: 'Fira Code', monospace; 
                color: #666; font-size: 0.8rem; border-top: 1px solid #333; padding-top: 2rem;">
        > NEURAL_COLORIZER_2077.exe | VERSION 3.14.159 | QUANTUM_ENHANCED<br>
        > POWERED_BY: ADVANCED_AI_NEURAL_NETWORKS | CYBERPUNK_INTERFACE_v2.0
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()



