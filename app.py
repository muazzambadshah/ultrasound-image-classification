import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime

from config import APP_TITLE, ORGANS
from utils import load_models, preprocess_image, safe_predict_image, decode_prediction, predict_heart_tabular
from report import build_pdf_report

st.set_page_config(page_title=APP_TITLE, page_icon="🩺", layout="centered")
st.title(APP_TITLE)

models = load_models()

with st.expander("Exam details"):
    patient_id = st.text_input("Patient ID", "PAT-0001")
    examiner = st.text_input("Examiner", "Examiner Name")
    notes = st.text_area("Clinical notes (optional)")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

organ = st.selectbox("Select organ/model", ORGANS)
st.divider()

if organ == "Heart Disease":
    st.subheader("Heart disease risk input")
    age = st.number_input("Age", 1, 120, 54)
    sex = st.selectbox("Sex (0=female, 1=male)", [0, 1], index=1)
    cp = st.selectbox("Chest pain type (0-3)", [0, 1, 2, 3], index=1)
    trestbps = st.number_input("Resting BP", 50, 250, 130)
    chol = st.number_input("Cholesterol", 80, 700, 246)
    fbs = st.selectbox("Fasting blood sugar >120", [0, 1], index=0)
    restecg = st.selectbox("Rest ECG (0-2)", [0, 1, 2], index=1)
    thalach = st.number_input("Max heart rate", 60, 250, 150)
    exang = st.selectbox("Exercise-induced angina", [0, 1], index=0)
    oldpeak = st.number_input("ST depression", 0.0, 10.0, 1.0, step=0.1)
    slope = st.selectbox("Slope (0-2)", [0, 1, 2], index=2)
    ca = st.selectbox("Major vessels (0-3)", [0, 1, 2, 3], index=0)

    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca]], dtype=np.float32)

    if st.button("Predict"):
        prob, raw = predict_heart_tabular