import streamlit as st
import json
import os
import base64
from jinja2 import Template
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import pandas as pd
from io import BytesIO, StringIO
from docx import Document
from openai import OpenAI
from dotenv import load_dotenv
import uuid
import tempfile
import shutil
import matplotlib
matplotlib.use('Agg')  # Prevent tkinter errors

# Load API Key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

st.title("✅ AI-Powered Software Requirement Extractor + ISO 29148 Evaluator")

language_options = ["English", "German", "Spanish", "French", "Italian"]
language_code_map = {"English": "en", "German": "de", "Spanish": "es", "French": "fr", "Italian": "it"}
selected_language = st.selectbox("🌐 Document Language", language_options, index=0)

feedback_log = []

uploaded_file = st.file_uploader("📂 Upload a Software Requirement Document (PDF or DOCX)", type=["pdf", "docx"])


def translate_to_english(text, source_lang):
    if source_lang == "en":
        return text
    prompt = [
        {"role": "system", "content": f"Translate the following {source_lang} text to English."},
        {"role": "user", "content": text}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=prompt
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"Translation failed: {str(e)}")
        return text

def extract_text_from_pdf(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        return "\n".join([page.get_text("text") for page in doc])
    except Exception as e:
        st.error(f"⚠️ Error extracting text from PDF: {str(e)}")
        return ""

def extract_text_from_docx(docx_file):
    try:
        doc = Document(docx_file)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"⚠️ Error extracting text from DOCX: {str(e)}")
        return ""

def split_text(text, chunk_size=6000, overlap=1000):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

def annotate_text(text, requirements):
    annotated = text
    for req in requirements:
        snippet = req["Requirement"]
        if snippet in annotated:
            style = "background-color: #cf2cff; color: black; padding: 2px 4px; border-radius: 6px; font-weight: bold;"
            if req["RequirementID"].startswith("NFR"):
                style = "background-color: #fbd4e6; color: black; padding: 2px 4px; border-radius: 6px; font-weight: bold;"
            annotated = annotated.replace(snippet, f'<span style="{style}">{snippet}</span>')
    return annotated

def get_requirement_extraction_prompt_json(chunk_text):
    example = {
        "functional_requirements": [
            {"RequirementID": "FR-001", "Requirement": "The system shall allow users to upload PDF files."}
        ],
        "non_functional_requirements": [
            {"RequirementID": "NFR-001", "Requirement": "The system shall respond to upload requests within 2 seconds."}
        ],
        "missing_critical_requirements": [
            "The document does not mention any backup or recovery mechanism."
        ],
        "recommendations": [
            "Add explicit performance thresholds for critical operations."
        ]
    }

    json_prompt = {
        "task": "Extract software requirements from a document.",
        "instructions": {
            "description": (
                "You are an AI expert in software requirements engineering. Your task is to extract only the functional and non-functional requirements "
                "from the provided document content. Do not include introductions, project background, context, definitions, or any non-requirement content."
            ),
            "output_format": example,
            "notes": [
                "Output must be in valid JSON format.",
                "Only include requirements, not descriptions or commentary.",
                "Do not include the same requirement more than once.",
                "Ensure each requirement is clearly worded and testable."
            ],
            "example_input_chunk": (
                "The system will allow the user to upload and store PDF files. Performance should be responsive, with no noticeable delay. "
                "There is currently no mention of backup and recovery. It is recommended to include scalability goals as well."
            ),
            "example_output": example
        },
        "document_chunk": chunk_text
    }

    return [
        {
            "role": "user",
            "content": f"Please extract software requirements using the following JSON prompt:\n\n{json.dumps(json_prompt, indent=2)}"
        }
    ]


def evaluate_requirement_quality_iso(requirement_text, req_id):
    example_input = {
        "RequirementID": "REQ-1",
        "RequirementText": "The user can split only one document at a time."
    }

    example_output = {
        "RequirementID": "REQ-1",
        "RequirementText": "The user can split only one document at a time.",
        "ISO29148_QualityAssessment": {
            "Appropriate": {"value": "Yes", "reason": "It aligns with user needs for document splitting."},
            "Complete": {"value": "Yes", "reason": "The description fully explains the behavior."},
            "Conforming": {"value": "Yes", "reason": "Follows expected structure for functional requirements."},
            "Correct": {"value": "Yes", "reason": "Describes intended system behavior accurately."},
            "Feasible": {"value": "Yes", "reason": "Technically achievable with standard tools."},
            "Necessary": {"value": "Yes", "reason": "A basic feature of a PDF splitting tool."},
            "Singular": {"value": "Yes", "reason": "Describes a single function without conjunctions."},
            "Unambiguous": {"value": "Yes", "reason": "Clearly states what the user can do."},
            "Verifiable": {"value": "Yes", "reason": "Can be tested with a single PDF document."}
        }
    }

    json_prompt = {
        "task": "Evaluate a software requirement using ISO/IEC 29148 quality characteristics.",
        "instructions": {
            "criteria": [
                "Appropriate", "Complete", "Conforming", "Correct",
                "Feasible", "Necessary", "Singular", "Unambiguous", "Verifiable"
            ],
            "output_format": example_output,
            "notes": [
                "Each criterion must be rated as 'Yes', 'No', or 'Partially'.",
                "Include a brief reason for each evaluation.",
                "Follow the output format exactly as shown in the example."
            ],
            "example_input": example_input,
            "example_output": example_output
        },
        "requirement_to_evaluate": {
            "RequirementID": req_id,
            "RequirementText": requirement_text
        }
    }

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": f"Please evaluate this requirement using the following JSON-based instruction:\n\n{json.dumps(json_prompt, indent=2)}"
                }
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"RequirementID": req_id, "error": str(e)}


def export_to_excel(requirements):
    rows = []
    for req in requirements:
        row = {"RequirementID": req["RequirementID"], "Requirement": req["Requirement"]}
        qa = req.get("quality_evaluation", {})
        for dim, val in qa.items():
            row[f"{dim}_Value"] = val["value"]
            row[f"{dim}_Reason"] = val["reason"]
        rows.append(row)
    return pd.DataFrame(rows)

def show_summary_metrics(requirements):
    fr_count = sum(1 for req in requirements if req["RequirementID"].startswith("FR"))
    nfr_count = sum(1 for req in requirements if req["RequirementID"].startswith("NFR"))
    other_count = len(requirements) - fr_count - nfr_count
    total = len(requirements)

    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Total Requirements", total)
    col2.metric("🛠 Functional Requirements", fr_count)
    col3.metric("🎯 Non-Functional Requirements", nfr_count)

    if other_count:
        st.warning(f"⚠️ {other_count} requirement(s) did not follow 'FR' or 'NFR' prefix convention.")


def show_summary_pie_chart(requirements):
    values = Counter()
    for req in requirements:
        qa = req.get("quality_evaluation", {})
        for _, dim in qa.items():
            values[dim["value"]] += 1
    if not values:
        st.warning("No quality evaluation data to display.")
        return

    labels = list(values.keys())
    counts = [values[l] for l in labels]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140)
    ax.set_title("ISO 29148 Evaluation Distribution")
    st.pyplot(fig)

if 'final_response' in st.session_state and st.session_state.get('evaluation_done'):
    all_reqs = st.session_state['final_response']['functional_requirements'] + st.session_state['final_response']['non_functional_requirements']
    if any("quality_evaluation" in req for req in all_reqs):
        st.subheader("📈 Evaluation Summary Dashboard")
        show_summary_metrics(all_reqs)
        show_summary_pie_chart(all_reqs)
    else:
        st.info("ℹ️ Quality evaluations not found. Please extract and evaluate requirements first.")

def show_evaluation_chart(requirements):
    quality_counts = defaultdict(lambda: {"Yes": 0, "Partially": 0, "No": 0})
    for req in requirements:
        for charac, result in req.get("quality_evaluation", {}).items():
            quality_counts[charac][result["value"]] += 1

    labels = list(quality_counts.keys())
    yes = [quality_counts[label]["Yes"] for label in labels]
    partially = [quality_counts[label]["Partially"] for label in labels]
    no = [quality_counts[label]["No"] for label in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    bar_width = 0.25
    x = range(len(labels))

    ax.bar(x, yes, width=bar_width, label='Yes')
    ax.bar([p + bar_width for p in x], partially, width=bar_width, label='Partially')
    ax.bar([p + 2 * bar_width for p in x], no, width=bar_width, label='No')

    ax.set_xticks([p + bar_width for p in x])
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Count")
    ax.set_title("ISO 29148 Evaluation Summary")
    ax.legend()

    st.pyplot(fig)


if uploaded_file:
    extracted_text = ""
    if uploaded_file.name.endswith(".pdf"):
        extracted_text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        extracted_text = extract_text_from_docx(uploaded_file)
    else:
        st.warning("Unsupported file type.")

    source_lang_code = language_code_map[selected_language]
    extracted_text = translate_to_english(extracted_text, source_lang_code)

    if st.toggle("🖍️ Show Annotated Document", value=True):
        st.session_state.show_annotated = True
    else:
        st.session_state.show_annotated = False

    if st.button("Extract & Evaluate Requirements"):
        if not extracted_text:
            st.warning("⚠️ No text found to analyze.")
        else:
            st.info("📤 Sending document to OpenAI for requirement extraction...")
            chunks = split_text(extracted_text)
            all_results = []

            for idx, chunk in enumerate(chunks):
                st.write(f"🚀 Processing chunk {idx + 1}/{len(chunks)}")
                prompt = get_requirement_extraction_prompt_json(chunk)

                try:
                    response = client.chat.completions.create(
                        model="gpt-4-turbo",
                        messages=prompt,
                        response_format={"type": "json_object"}
                    )
                    data = json.loads(response.choices[0].message.content)
                    all_results.append(data)
                except Exception as e:
                    st.warning(f"⚠️ Error in chunk {idx + 1}: {str(e)}")

            final_response = {
                "functional_requirements": [],
                "non_functional_requirements": [],
                "missing_critical_requirements": [],
                "recommendations": []
            }

            for res in all_results:
                final_response["functional_requirements"].extend(res.get("functional_requirements", []))
                final_response["non_functional_requirements"].extend(res.get("non_functional_requirements", []))
                final_response["missing_critical_requirements"].extend(res.get("missing_critical_requirements", []))
                final_response["recommendations"].extend(res.get("recommendations", []))

            all_reqs = final_response["functional_requirements"] + final_response["non_functional_requirements"]

            if st.session_state.get("show_annotated"):
                st.markdown(annotate_text(extracted_text, all_reqs), unsafe_allow_html=True)
            else:
                st.text_area("📄 Extracted Full Document Content (Editable):", extracted_text, height=300)

            st.subheader("📋 Extracted Requirements")
            st.markdown("### 🛠 Functional Requirements")
            for req in sorted(final_response["functional_requirements"], key=lambda r: r["RequirementID"]):
                st.markdown(f"- **{req['RequirementID']}**: {req['Requirement']}")

            st.markdown("### 🎯 Non-Functional Requirements")
            for req in sorted(final_response["non_functional_requirements"], key=lambda r: r["RequirementID"]):
                st.markdown(f"- **{req['RequirementID']}**: {req['Requirement']}")

            st.subheader("🧪 ISO 29148 Quality Evaluation")

            for req in all_reqs:
                with st.expander(f"🔍 {req['RequirementID']} Evaluation", expanded=False):
                    with st.spinner("Evaluating..."):
                        result = evaluate_requirement_quality_iso(req["Requirement"], req["RequirementID"])
                        req["quality_evaluation"] = result.get("ISO29148_QualityAssessment", {})
                        for dim, val in req["quality_evaluation"].items():
                            st.markdown(f"• **{dim}**: {val['value']} — _{val['reason']}_")

            show_summary_metrics(all_reqs)
            show_evaluation_chart(all_reqs)
            show_summary_pie_chart(all_reqs)

            json_report = json.dumps(final_response, indent=4)
            st.download_button("📥 Download JSON Report", json_report, "requirement_analysis.json", "application/json")

            excel_df = export_to_excel(all_reqs)
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                excel_df.to_excel(writer, index=False)
            excel_buffer.seek(0)
            st.download_button("📊 Download Excel Report", data=excel_buffer, file_name="requirement_analysis.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            st.download_button("📄 Download CSV Report", data=excel_df.to_csv(index=False), file_name="requirement_analysis.csv", mime="text/csv")

            st.subheader("🗣️ User Feedback")
            user_feedback = st.text_area("💬 Leave your feedback about this analysis:")
            if st.button("✅ Submit Feedback"):
                feedback_entry = {
                    "filename": uploaded_file.name,
                    "feedback": user_feedback,
                    "total_requirements": len(all_reqs)
                }
                feedback_log.append(feedback_entry)
                st.success("Thank you for your feedback!")

            if feedback_log:
                feedback_json = json.dumps(feedback_log, indent=2)
                st.download_button("📤 Download Feedback Log", data=feedback_json, file_name="feedback_log.json", mime="application/json")

