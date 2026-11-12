from fpdf import FPDF
import io

def build_pdf_report(patient_id, examiner, organ, prediction_label,
                     confidence, timestamp, clinical_notes, image=None, extra_details=""):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Ultrasound Diagnostic Report", ln=True, align="C")

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Patient ID: {patient_id}", ln=True)
    pdf.cell(0, 8, f"Examiner: {examiner}", ln=True)
    pdf.cell(0, 8, f"Organ/Exam: {organ}", ln=True)
    pdf.cell(0, 8, f"Timestamp: {timestamp}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Result", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, f"Prediction: {prediction_label}\nConfidence: {confidence:.2f}")

    if extra_details:
        pdf.ln(3)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Details", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 8, extra_details)

    if clinical_notes:
        pdf.ln(3)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Clinical Notes", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 8, clinical_notes)

    if image is not None:
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Image Preview", ln=True)
        img_stream = io.BytesIO()
        image.save(img_stream, format="PNG")
        img_stream.seek(0)
        pdf.image(img_stream, x=15, y=None, w=180)

    pdf.ln(10)
    pdf.set_font("Arial", "I", 9)
    pdf.multi_cell(0, 6,
        "Note: This AI-generated result is for research/educational use only. "
        "It must be corroborated by a qualified clinician."
    )

    out = io.BytesIO()
    pdf.output(out)
    out.seek(0)
    return out.read()