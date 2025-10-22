#Title: “Multiclass Fish Image Classification Using Deep Learning”

Subtitle:
Developed by Edison Xavier

#Details:

Domain: Computer Vision

Framework: TensorFlow & Keras

Deployment: Streamlit

Date: October 2025

#Objective:
To classify fish species from images using Convolutional Neural Networks (CNN) and pre-trained deep learning models.

Dataset:

Folder structure: train/, val/, test/

Images of multiple fish species

Total Classes: 11 fish categories

Goal:
Build a robust model, compare multiple architectures, and deploy a real-time classification app.

#Methodology Pipeline

Dataset (.zip)
     ↓
Data Preprocessing & Augmentation
     ↓
Model Training (CNN + Pretrained Models)
     ↓
Model Evaluation & Comparison
     ↓
Best Model Saved (.pkl)
     ↓
Deployment via Streamlit

#Data Preprocessing & Augmentation

Steps:

Rescaled all images to [0, 1] range

Applied augmentation:

Rotation (±20°)

Zoom (20%)

Horizontal Flip

Image Size: 224×224

Batch Size: 32

#Model Training

Approach 1 – Custom CNN:
Built from scratch using 3 Conv2D + MaxPooling layers with Dropout.

Approach 2 – Transfer Learning:
Fine-tuned these pretrained models:

VGG16

ResNet50

MobileNet

InceptionV3

EfficientNetB0 (optional/fallback to weights=None)

Training Details:

Optimizer: Adam (lr=0.0001)

Loss: Categorical Crossentropy

Epochs: 10

#Model Evaluation Metrics

Metrics Computed:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

#Confusion Matrix & Plots

Include screenshots or sample visuals:

Confusion Matrix heatmap

Accuracy vs Epochs (Training/Validation)

Loss vs Epochs plots

Interpretation:

The model shows clear separation between fish categories.

Minor confusion among similar species (e.g., sea_bass vs red_sea_bream).

#Model Saving

Saved File:
best_fish_model.pkl

Description:

Serialized using pickle.dump()

Contains architecture + trained weights

Reused during Streamlit deployment

#Streamlit Deployment

Goal:
Allow users to upload fish images and get real-time predictions.

Features:

Upload .jpg / .png fish image

Predict species instantly

Display confidence scores as a bar chart

App Output:
🎯 Predicted: Red Mullet
📊 Confidence: 96.8%

#Project Architecture (End-to-End)

Dataset (.zip)
  ↓
Data Generator
  ↓
Model Training (CNN + Transfer Learning)
  ↓
Evaluation
  ↓
Best Model (.pkl)
  ↓
Streamlit App
  ↓
User Upload → Prediction + Confidence

#Results Summary

Best Model: ResNet50
Accuracy: ~93%
Loss: 0.32
Deployment: Streamlit Web App

✅ Real-time predictions
✅ Confusion matrix visualizations
✅ Saved reusable .pkl model

#Conclusion

Deep learning (transfer learning) significantly improved accuracy.

ResNet50 achieved best results for multi-class fish classification.

Successfully deployed a user-interactive web app for predictions.

Future Work:

Add support for live webcam detection

Use EfficientNetB7 or Vision Transformers

Deploy on cloud (AWS / Hugging Face Spaces)

#Tools & Libraries

| Category      | Tools Used                                         |
| ------------- | -------------------------------------------------- |
| Programming   | Python                                             |
| Libraries     | TensorFlow, Keras, NumPy, Matplotlib, Scikit-learn |
| Deployment    | Streamlit                                          |
| Visualization | Matplotlib, Streamlit Charts                       |
| Serialization | Pickle                                             |


#References

TensorFlow Documentation

Keras Model Zoo

Streamlit Docs

Fish Species Open Image Dataset

#Thank You

Developed by Edison Xavier
Deep Learning & Computer Vision Project
🐠 Fish Image Classification App


