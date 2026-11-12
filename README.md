# Multi-organ Ultrasound Diagnostics

CNN-based ultrasound image classifier covering abdominal, breast, kidney and liver scans (Normal vs Abnormal), plus a heart-disease tabular model and automatic PDF report generation.

## Features
- Separate fine-tuned CNN per organ (224x224 input)
- Heart-disease risk prediction from tabular data
- Downloadable PDF diagnostic report
- Streamlit web interface

## Tech Stack
Python, TensorFlow 2.10/Keras, Scikit-learn, Pillow, Streamlit, ReportLab

## Getting Started
```bash
git clone https://github.com/muazzambadshah/ultrasound-image-classification.git
cd ultrasound-image-classification
python -m venv venv
venv\Scripts\activate        # Windows  (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

> **Note:** Trained model weights and datasets are not included in this repository (size). Place the `.keras` / `.pkl` files listed in `config.py` inside `models/` before running.

## Screenshots
_Screenshots coming soon._

## Author
**Muazzam Badshah** - AI Researcher, NCAI (NUST), Islamabad
[Portfolio](https://muazzam-portfolio-2wk1.vercel.app/) | [LinkedIn](https://www.linkedin.com/in/muazzam-badshah-338881381) | [GitHub](https://github.com/muazzambadshah)
