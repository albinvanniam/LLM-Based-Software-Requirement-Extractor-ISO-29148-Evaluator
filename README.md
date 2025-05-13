# LLM-Based Software Requirement Extractor & ISO 29148 Evaluator

This project is a LLM-driven tool designed to assist software analysts by **automatically extracting requirements** from specification documents and **evaluating them against ISO/IEC/IEEE 29148 standards**. Built using **OpenAI's GPT-4 Turbo** and integrated into a **Streamlit web app**, it ensures high-quality, structured software requirement analysis and guidance.

##  Features

-  **Automated Requirement Extraction**
  - Supports both **Functional Requirements (FR)** and **Non-Functional Requirements (NFR)**
  - Input formats: `.pdf` (via PyMuPDF) and `.docx` (via python-docx)

-  **LLM-Based Evaluation**
  - Uses OpenAI GPT-4 Turbo with structured JSON and one-shot prompting
  - Evaluates requirements against ISO/IEC/IEEE 29148 quality characteristics
  - Provides judgments: **Yes / No / Partially**, with reasoning

-  **Multi-Format Export**
  - Outputs results in **JSON**, **CSV**, and **Excel (.xlsx)** using Pandas and OpenPyXL

- **Improvement Suggestions**
  - For any requirement evaluated as "No" or "Partially", the tool suggests revisions

-  **Future: Visual Dashboard**
  - Planned feature to show pie charts, bar charts, and summary stats for evaluations

##  Demo Preview

> A live demo is integrated into the Streamlit app (details to be added once deployed).

##  Getting Started

### 1. Clone the Repository


git clone https://github.com/yourusername/llm-requirements-tool.git
cd llm-requirements-tool

### 2. Set Up Environment


python -m venv venv
venv\Scripts\activate  # On Windows
# or
source venv/bin/activate  # On Unix/macOS

pip install -r requirements.txt

### 3.Configure Environment Variables
Create a .env file (make sure it's in .gitignore) and include your OpenAI API key:


OPENAI_API_KEY=your-api-key-here


### 4. Run the App

streamlit run app.py



