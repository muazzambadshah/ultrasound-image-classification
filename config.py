APP_TITLE = "Multi-organ Ultrasound Diagnostics"

INPUT_SIZE = (224, 224)

MODEL_PATHS = {
    "Abdominal": "models/abdominal_model.keras",
    "Breast": "models/breast_model.keras",
    "Kidney": "models/kidney_ultrasound_model.keras",
    "Liver": "models/liver_model_finetuned.keras",
    "Heart Disease": "models/heart_disease_model.pkl",
}

LABEL_MAP = {
    "Abdominal": {0: "Normal", 1: "Abnormal"},
    "Breast": {0: "Normal", 1: "Abnormal"},
    "Kidney": {0: "Normal", 1: "Abnormal"},
    "Liver": {0: "Normal", 1: "Abnormal"},
    "Heart Disease": {0: "No disease", 1: "Disease"},
}

ORGANS = list(MODEL_PATHS.keys())