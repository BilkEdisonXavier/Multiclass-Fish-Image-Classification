import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from PIL import Image


# Load the Trained Model

st.set_page_config(page_title="🐠 Fish Species Classifier", page_icon="🐟", layout="centered")

@st.cache_resource
def load_model():
    with open("best_fish_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Class Labels (Update if needed)

class_labels = [
    "animal fish",
    "animal fish bass",
    "fish sea_food black_sea_sprat",
    "fish sea_food gilt_head_bream",
    "fish sea_food hourse_mackerel",
    "fish sea_food red_mullet",
    "fish sea_food red_sea_bream",
    "fish sea_food sea_bass",
    "fish sea_food shrimp",
    "fish sea_food striped_red_mullet",
    "fish sea_food trout"
]

# Streamlit App UI

st.title("🐠 Fish Classification App")
st.markdown("""
Upload an image of a fish 🐟 and the app will predict its species  
based on your trained deep learning model.
""")

uploaded_file = st.file_uploader("📤 Upload a fish image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image_pil = Image.open(uploaded_file).convert("RGB")
    st.image(image_pil, caption="Uploaded Image", use_container_width=True)

    # Preprocess Image
    
    img = image_pil.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array / 255.0)

    # Prediction
    
    try:
        preds = model.predict(img_array)
        confidence_scores = np.max(preds, axis=1)
        predicted_class = np.argmax(preds, axis=1)[0]
        predicted_label = class_labels[predicted_class]

        st.subheader("🎯 Prediction Result")
        st.success(f"Predicted Fish: **{predicted_label}**")
        st.info(f"Model Confidence: **{confidence_scores[0]*100:.2f}%**")

        # Display confidence chart
        st.subheader("📊 Confidence Scores")
        confidence_df = {
            "Class": class_labels,
            "Confidence": preds[0]
        }
        st.bar_chart(data=dict(zip(class_labels, preds[0])))

    except Exception as e:
        st.error(f"⚠️ Prediction Error: {e}")
else:
    st.info("👆 Upload a fish image to start prediction.")
