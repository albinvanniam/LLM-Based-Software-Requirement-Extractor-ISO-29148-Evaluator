from flask import Flask, request, jsonify
import os
import pdfplumber
from docx import Document
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI()

app = Flask(__name__)

# Home Route (MOVE THIS UP)
@app.route("/")
def home():
    return "Flask API is running! Use the /upload endpoint."

# Extract text from PDF
def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Extract text from DOCX
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

# Generate requirements using LLM
def generate_requirements(document_text, model="gpt-3.5-turbo-1106"):
    prompt = f"""
    Extract all functional and non-functional requirements from this document:
    {document_text}
    """

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# API Route for File Upload
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    file_path = f"/tmp/{file.filename}"
    file.save(file_path)

    if file.filename.endswith(".pdf"):
        document_text = extract_text_from_pdf(file_path)
    elif file.filename.endswith(".docx"):
        document_text = extract_text_from_docx(file_path)
    elif file.filename.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            document_text = f.read()
    else:
        return jsonify({"error": "Unsupported file format. Upload PDF, DOCX, or TXT."}), 400

    requirements = generate_requirements(document_text)

    return jsonify({"requirements": requirements})

# Run the Flask App
if __name__ == '__main__':
    app.run(debug=True)