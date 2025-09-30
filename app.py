import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# Page configuration
st.set_page_config(
    page_title="Tomato Plant Disease Detection",
    page_icon="🍅",
    layout="wide"
)

# Model paths
MODEL_CUSTOM = "models/model_custom.keras"
MODEL_TRANSFER = "models/model_transfer.keras"
CLASS_JSON = "class_names.json"

@st.cache_resource
def load_models():
    """Load both trained models and class names"""
    if not os.path.exists(MODEL_CUSTOM):
        st.error(f"Custom CNN model not found at: {MODEL_CUSTOM}")
        st.info("Please train the model using notebook 02_model_custom_cnn.ipynb")
        return None, None, None
    
    if not os.path.exists(MODEL_TRANSFER):
        st.error(f"Transfer Learning model not found at: {MODEL_TRANSFER}")
        st.info("Please train the model using notebook 03_model_transfer_learning.ipynb")
        return None, None, None
    
    if not os.path.exists(CLASS_JSON):
        st.error(f"Class names file not found at: {CLASS_JSON}")
        return None, None, None
    
    try:
        model_custom = tf.keras.models.load_model(MODEL_CUSTOM)
        model_transfer = tf.keras.models.load_model(MODEL_TRANSFER)
        
        with open(CLASS_JSON, "r") as f:
            class_names = json.load(f)
        
        return model_custom, model_transfer, class_names
    
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

def preprocess_image(image, img_size=(224, 224)):
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    image = image.resize(img_size)
    
    # Convert to array
    img_array = np.array(image)
    
    # Expand dimensions to match model input
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def get_disease_info(disease_name):
    
    disease_info = {
        "Tomato_Early_blight": {
            "description": "Early blight is a common tomato disease caused by the fungus Alternaria solani.",
            "symptoms": [
                "Dark spots with target-like rings on leaves",
                "Lower leaves affected first",
                "Yellow tissue around spots",
                "Leaves may drop prematurely"
            ],
            "treatment": [
                "Remove infected leaves immediately",
                "Apply appropriate fungicide",
                "Improve air circulation around plants",
                "Avoid overhead watering",
                "Rotate crops yearly"
            ],
            "severity": "Medium"
        },
        "Tomato_healthy": {
            "description": "The plant appears healthy with no visible signs of disease.",
            "symptoms": [
                "Green, vibrant leaves",
                "No spots or discoloration",
                "Normal growth pattern",
                "Healthy stem structure"
            ],
            "treatment": [
                "Continue regular care and monitoring",
                "Maintain proper watering schedule",
                "Ensure adequate nutrition",
                "Monitor for early signs of disease"
            ],
            "severity": "None"
        },
        "Tomato_Late_blight": {
            "description": "Late blight is a serious disease caused by Phytophthora infestans, the same pathogen that caused the Irish potato famine.",
            "symptoms": [
                "Water-soaked spots on leaves",
                "White fuzzy growth on leaf undersides",
                "Brown lesions on stems",
                "Rapid spread in humid conditions",
                "Fruit rot with brown patches"
            ],
            "treatment": [
                "Remove and destroy infected plants immediately",
                "Apply copper-based fungicide preventatively",
                "Improve air circulation",
                "Avoid watering late in the day",
                "Consider resistant varieties"
            ],
            "severity": "High"
        }
    }
    
    return disease_info.get(disease_name, {
        "description": "Information not available",
        "symptoms": ["Unknown"],
        "treatment": ["Consult an agricultural expert"],
        "severity": "Unknown"
    })

# Load models
model_custom, model_transfer, class_names = load_models()

if model_custom is None or model_transfer is None:
    st.stop()

# App header
st.title("Tomato Plant Disease Detection System")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("About")
    st.write("""
    This application uses deep learning to detect diseases in tomato plants.
    
    **Two Models:**
    - Custom CNN: Built from scratch
    - Transfer Learning: MobileNetV2 pre-trained
    
    **Detectable Diseases:**
    - Early Blight
    - Late Blight
    - Healthy Plants
    """)
    
    st.markdown("---")
    st.subheader("How to Use")
    st.write("""
    1. Upload a tomato leaf image
    2. Click 'Analyze Image'
    3. View predictions from both models
    4. Read disease information and treatment
    """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a tomato leaf image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of a tomato leaf"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.subheader("Analysis Results")
    
    if uploaded_file is not None:
        if st.button("Analyze Image", type="primary"):
            with st.spinner("Analyzing image..."):
                # Preprocess image
                img_array = preprocess_image(image)
                
                # Get predictions from both models
                pred_custom = model_custom.predict(img_array, verbose=0)
                pred_transfer = model_transfer.predict(img_array, verbose=0)
                
                # Get predicted classes and confidences
                idx_custom = int(np.argmax(pred_custom))
                idx_transfer = int(np.argmax(pred_transfer))
                
                conf_custom = float(np.max(pred_custom)) * 100
                conf_transfer = float(np.max(pred_transfer)) * 100
                
                class_custom = class_names[idx_custom]
                class_transfer = class_names[idx_transfer]
                
                # Display results
                st.success("Analysis Complete!")
                
                # Model predictions
                st.markdown("### Model Predictions")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.metric(
                        label="Custom CNN",
                        value=class_custom.replace("_", " ").title(),
                        delta=f"{conf_custom:.1f}% confidence"
                    )
                
                with col_b:
                    st.metric(
                        label="Transfer Learning",
                        value=class_transfer.replace("_", " ").title(),
                        delta=f"{conf_transfer:.1f}% confidence"
                    )
                
                # Show all class probabilities
                with st.expander("View Detailed Probabilities"):
                    st.markdown("**Custom CNN Probabilities:**")
                    for i, cls in enumerate(class_names):
                        prob = pred_custom[0][i] * 100
                        st.write(f"- {cls.replace('_', ' ').title()}: {prob:.2f}%")
                    
                    st.markdown("**Transfer Learning Probabilities:**")
                    for i, cls in enumerate(class_names):
                        prob = pred_transfer[0][i] * 100
                        st.write(f"- {cls.replace('_', ' ').title()}: {prob:.2f}%")
    else:
        st.info("Upload an image to get started")

# Disease Information Section
if uploaded_file is not None and 'class_transfer' in locals():
    st.markdown("---")
    st.subheader("Disease Information")
    
    # Use transfer learning prediction
    disease_info = get_disease_info(class_transfer)
    
    # Severity indicator
    severity_colors = {
        "None": "🟢",
        "Low": "🟡",
        "Medium": "🟠",
        "High": "🔴",
        "Unknown": "⚪"
    }
    
    st.markdown(f"**Severity Level:** {severity_colors.get(disease_info['severity'], '⚪')} {disease_info['severity']}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Description:**")
        st.write(disease_info['description'])
        
        st.markdown("**Symptoms:**")
        for symptom in disease_info['symptoms']:
            st.write(f"- {symptom}")
    
    with col2:
        st.markdown("**Recommended Treatment:**")
        for treatment in disease_info['treatment']:
            st.write(f"- {treatment}")

# Footer
st.markdown("---")
st.caption("Tomato Plant Disease Detection System | Computer Vision CW Part B")