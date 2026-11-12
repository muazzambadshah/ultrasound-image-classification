import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
from PIL import Image
from config import MODEL_PATHS, INPUT_SIZE, LABEL_MAP


# Cache models so they load only once
@st.cache(allow_output_mutation=True)
def load_models():
    models = {}
    for organ, path in MODEL_PATHS.items():
        if path.endswith(".keras"):
            # compile=False avoids InputLayer/batch_shape deserialization issues
            models[organ] = tf.keras.models.load_model(path, compile=False)
        elif path.endswith(".pkl"):
            models[organ] = joblib.load(path)
    return models


def preprocess_image(image: Image.Image, target_size=(224, 224)):
    """Resize and normalize an image for model input."""
    img = image.resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:  # grayscale → RGB
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:  # RGBA → RGB
        arr = arr[..., :3]
    arr = np.expand_dims(arr, axis=0)
    return arr


def safe_predict_image(model, img_batch):
    """Run prediction and return probabilities in a consistent format."""
    raw = model.predict(img_batch)
    if raw.ndim == 2 and raw.shape[1] > 1:
        prob = raw[0]
    else:
        p = float(raw[0] if raw.ndim == 1 else raw[0][0])
        prob = np.array([1.0 - p, p], dtype=np.float32)
    return prob, raw


def decode_prediction(organ, prob):
    """Convert model output into label, confidence, and class index."""
    idx = int(np.argmax(prob))
    confidence = float(prob[idx])
    label = LABEL_MAP.get(organ, {}).get(idx, f"Class {idx}")
    return label, confidence, idx


def predict_heart_tabular(model, features):
    """Handle tabular heart model predictions (sklearn or similar)."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)
        prob = np.array(proba[0], dtype=np.float32)
        raw = proba
    else:
        pred = model.predict(features)
        y = int(pred[0])
        prob = np.array([1 - y, y], dtype=np.float32)
        raw = pred
    return prob, raw