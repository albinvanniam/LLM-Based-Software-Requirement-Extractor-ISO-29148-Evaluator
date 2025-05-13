import streamlit as st
import os
import pdfplumber
import pandas as pd
from docx import Document
from io import BytesIO
import pdfkit
from dotenv import load_dotenv
from utils.processing import extract_text, generate_requirements, validate_requirements, prioritize_requirements

def app():
    st.title("✅ Quality Check")
    st.write("This is the quality check page.")




# Load API Key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.title("📄 LLM-Powered Requirements Extraction Tool")
st.sidebar.header("Upload Project Document")
file = st.sidebar.file_uploader("Choose a file", type=["pdf", "docx", "txt"])

if file:
    st.sidebar.success("File uploaded successfully!")
    text = extract_text(file)

    if text:
        st.subheader("Extracted Requirements:")
        requirements = generate_requirements(text)
        edited_text = st.text_area("Edit Requirements", requirements, height=300)

        # Validation
        st.subheader("Requirement Validation:")
        validation = validate_requirements(edited_text)
        st.write(validation)

        # Prioritization
        st.subheader("Requirement Prioritization:")
        priority = prioritize_requirements(edited_text)
        st.write(priority)

        # Export Options
        st.sidebar.subheader("Download Options")

        def export_docx(text):
            doc = Document()
            doc.add_paragraph(text)
            buffer = BytesIO()
            doc.save(buffer)
            buffer.seek(0)
            return buffer

        def export_csv(text):
            df = pd.DataFrame({"Requirements": text.split('\n')})
            buffer = BytesIO()
            df.to_csv(buffer, index=False)
            buffer.seek(0)
            return buffer

        def export_pdf(text):
            pdf_path = "requirements.pdf"
            config = pdfkit.configuration(wkhtmltopdf="C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe")
            pdfkit.from_string(text, pdf_path, configuration=config)
            with open(pdf_path, "rb") as f:
                return f.read()

        st.sidebar.download_button("Download as DOCX", export_docx(edited_text), "requirements.docx")
        st.sidebar.download_button("Download as CSV", export_csv(edited_text), "requirements.csv")
        st.sidebar.download_button("Download as PDF", export_pdf(edited_text), "requirements.pdf", "application/pdf")
    else:
        st.sidebar.error("Unsupported file format.")
