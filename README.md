# plant-disease-detection

## Overview
Deep learning system for detecting tomato plant diseases using two different models: Custom CNN and Transfer Learning with MobileNetV2.

## Dataset
- Source: PlantVillage
- Classes: Tomato Early Blight, Tomato Healthy, Tomato Late Blight
- Total images: 3,000 (1,000 per class)
- Split: 75% training, 25% validation

## Files Structure

plant_disease_project/
├── data/
│   ├── Tomato_Early_blight/
│   │   └── (1000 images)
│   ├── Tomato_healthy/
│   │   └── (1000 images)
│   └── Tomato_Late_blight/
│       └── (1000 images)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_custom_cnn.ipynb
│   └── 03_model_transfer_learning.ipynb
├── models/
│   ├── model_custom.keras
│   └── model_transfer.keras
├── app.py
├── class_names.json
├── requirements.txt
└── README.md

## Requirements
tensorflow==2.15.0
streamlit==1.50.0
pillow==11.3.0
numpy==2.3.3
matplotlib
scikit-learn
seaborn

## Installation
in the bash run :- pip install -r requirements.txt

## Training Models
Run notebooks in order:

01_data_exploration.ipynb - Verify dataset
02_model_custom_cnn.ipynb - Train custom CNN
03_model_transfer_learning.ipynb - Train transfer learning model

## Application
Running the Application :- streamlit run app.py

## Model Performance

Custom CNN: 87.3% accuracy
Transfer Learning: 95.7% accuracy

## Author
T.P.W.B. Samod - YR4COBSCCOMP232P-037