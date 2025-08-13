# see.py

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

@st.cache_resource
def load_model():
    # 1) Recreate the exact same architecture
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    # 2) Load your trained weights
    model.load_weights('mobilenet_weights.weights.h5')

    # 3) Compile (if you’ll use `model.evaluate` or similar)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

model = load_model()

st.title("🐱 vs 🐶 Cat/Dog Classifier")

uploaded = st.file_uploader("Upload a cat or dog image", type=["jpg", "png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB").resize((224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)  # shape (1,224,224,3)
    pred = model.predict(arr)[0][0]

    label = "Dog 🐶" if pred > 0.5 else "Cat 🐱"
    conf  = pred if pred > 0.5 else 1 - pred

    st.write(f"**Prediction:** {label}")
    st.write(f"**Confidence:** {conf:.2%}")
