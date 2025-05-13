import pdfplumber
from docx import Document
import os
import openai
from dotenv import load_dotenv

# Load API Key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def extract_text(file):
    if file.type == "application/pdf":
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    elif file.type == "text/plain":
        return file.getvalue().decode("utf-8")
    else:
        return None

def generate_requirements(text):
    prompt = f"""
    Extract and categorize functional and non-functional requirements from this document.
    Then classify them into Security, Performance, UX, and Hardware requirements.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt + text}]
    )
    return response["choices"][0]["message"]["content"]

def validate_requirements(requirements):
    prompt = f"""Review the following software requirements and suggest missing, inconsistent, 
    or conflicting ones:
    {requirements}"""
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

def prioritize_requirements(requirements):
    prompt = f"""Categorize these software requirements into 'High Priority', 'Medium Priority', and 'Low Priority':
    {requirements}"""
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]
