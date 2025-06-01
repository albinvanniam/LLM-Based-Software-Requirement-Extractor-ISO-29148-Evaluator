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
from difflib import SequenceMatcher
import re
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # For consistent results


matplotlib.use('Agg')  # Prevent tkinter errors

# Load API Key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

st.title(" AI-Powered Software Requirement Extractor + ISO 29148 Evaluator")

language_options = ["English", "German", "Spanish", "French", "Italian"]
language_code_map = {"English": "en", "German": "de", "Spanish": "es", "French": "fr", "Italian": "it"}
selected_language = st.selectbox("🌐 Document Language", language_options, index=0)

feedback_log = []

FEEDBACK_FILE = "feedback_log.json"

# Load existing feedback from file if exists
if os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        try:
            st.session_state["feedback_log"] = json.load(f)
        except json.JSONDecodeError:
            st.session_state["feedback_log"] = []
else:
    st.session_state["feedback_log"] = []

uploaded_file = st.file_uploader("💻 Upload a Software Requirement Document (PDF or DOCX)", type=["pdf", "docx"])


def translate_to_english(text, source_lang):
    try:
        detected_lang = detect(text)
    except Exception:
        detected_lang = "unknown"

    if detected_lang == "en":
        # Document is already in English; skip translation
        return text

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

def clean_text(text):
    # Merge broken lines and normalize spaces
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  # Join mid-sentence line breaks
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
    return text.strip()


def get_document_objectives_prompt(chunk_text):
    return [
        {
            "role": "system",
            "content": (
                "You are an expert requirements engineer. Given a chunk of a requirement document, "
                "identify high-level objectives, major goals, and overall purpose of the system being described. "
                "Avoid listing individual requirements. Focus only on what the document aims to achieve overall."
            )
        },
        {
            "role": "user",
            "content": f"Extract objectives, major goals, and purpose from the following text:\n\n{chunk_text}"
        }
    ]


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
                "You are an AI expert in software requirements engineering. Your task is to extract all functional and "
                "non-functional requirements from the provided document content. Do not include background, context, "
                "marketing, or explanatory text. Focus only on statements that describe system behavior, user needs, "
                "performance goals, environmental assumptions, or implementation constraints."
            ),
            "output_format": {
                "functional_requirements": [
                    {"RequirementID": "FR-001", "Requirement": "The system shall allow users to upload PDF files."}
                ],
                "non_functional_requirements": [
                    {
                        "RequirementID": "NFR-001",
                        "Requirement": "The system shall respond to upload requests within 2 seconds."
                    }
                ],
                "missing_critical_requirements": [
                    "The document does not mention any backup or recovery mechanism."
                ],
                "recommendations": [
                    "Add explicit performance thresholds for critical operations."
                ]
            },
            "notes": [
                "Output must be in valid JSON format.",
                (
                    "Only include actual requirements — statements that express system behavior, "
                    "conditions, or capabilities."
                ),
                "Ensure each requirement is atomic, clear, and testable.",
                (
                    "Include all requirement-like statements even if they are not explicitly labeled "
                    "with 'FR' or 'NFR'."
                ),
                "Include system-level, software-level, and communication-level behaviors.",
                (
                    "Identify requirements in structured formats such as '3.2.x.y', bulleted lists, or "
                    "clauses using 'shall', 'must', 'is required to', or 'will'."
                ),
                (
                    "Extract both capabilities (functional) and constraints, assumptions, compatibility "
                    "notes (non-functional)."
                ),
                "Recognize requirements written as objectives, features, or stakeholder goals.",
                (
                    "Do not ignore implied behaviors even if they are not phrased as mandatory "
                    "(e.g., 'should', 'includes')."
                )
            ],
            "example_input_chunk": (
                "3.2.1.1 agentMom shall support the ability to send unicast message.\n"
                "3.2.1.3 Unicast message shall only be received by the specified address.\n"
                "5.1 The website should maintain rapid response time suitable for high school students.\n"
                "The site should allow searches for internal and external CS information.\n"
                "3.13 Consider a CS Minor — present interdisciplinary value of combining CS with law or medicine.\n"
                "Constraints include bandwidth limitations and Chancellor’s Office resource availability.\n"
                "The system should track hits and adapt design based on usage analytics.\n"
                "The system shall work on IE, Firefox, and Netscape browsers."
            ),
            "example_output": {
                "functional_requirements": [
                    {
                        "RequirementID": "FR-001",
                        "Requirement": "agentMom shall support the ability to send unicast message."
                    },
                    {
                        "RequirementID": "FR-002",
                        "Requirement": "Unicast message shall only be received by the specified address."
                    },
                    {
                        "RequirementID": "FR-003",
                        "Requirement": "The system shall allow searches for both internal and external" 
                                       "CS-related information."
                    },
                    {
                        "RequirementID": "FR-004",
                        "Requirement": "The system shall be compatible with IE, Firefox, and Netscape browsers."
                    },
                    {
                        "RequirementID": "FR-005",
                        "Requirement": "The system shall include a section to promote the value of a CS minor" 
                                       "in other disciplines."
                    }
                ],
                "non_functional_requirements": [
                    {
                        "RequirementID": "NFR-001",
                        "Requirement": "The website shall maintain fast response times to suit high school users."
                    },
                    {
                        "RequirementID": "NFR-002",
                        "Requirement": "The system shall track site usage metrics and adjust content" 
                                       "based on analytics."
                    },
                    {
                        "RequirementID": "NFR-003",
                        "Requirement": "The system shall operate under bandwidth and resource availability constraints."
                    }
                ],
                "missing_critical_requirements": [],
                "recommendations": []
            }
        },
        "document_chunk": chunk_text
    }


    return [
        {
            "role": "user",
            "content": f"Please extract software requirements using the following JSON prompt:\n\n{json.dumps(json_prompt, indent=2)}"
        }
    ]

def get_updated_iso_prompt(requirement_text, req_id):
    prompt = {
        "task": "Evaluate a software requirement using ISO/IEC 29148 quality characteristics with numeric scores.",
        "instructions": {
            "criteria": [
                "Appropriate", "Complete", "Conforming", "Correct",
                "Feasible", "Necessary", "Singular", "Unambiguous", "Verifiable"
            ],
            "scoring_scale": {
                "0": "Not satisfied at all",
                "1": "Poorly satisfied",
                "2": "Partially satisfied",
                "3": "Mostly satisfied",
                "4": "Fully satisfied"
            },
            "output_format": {
                "RequirementID": req_id,
                "RequirementText": requirement_text,
                "ISO29148_QualityAssessment": {
                    "Appropriate": {"score": 3, "justification": "Mostly satisfies user relevance."},
                    "Complete": {"score": 4, "justification": "Fully details the requirement."},
                    "Conforming": {"score": 3, "justification": "Follows expected structure."},
                    "Correct": {"score": 4, "justification": "Accurately describes behavior."},
                    "Feasible": {"score": 4, "justification": "Technically realistic with available tools."},
                    "Necessary": {"score": 3, "justification": "Important but not critical."},
                    "Singular": {"score": 4, "justification": "Describes a single behavior."},
                    "Unambiguous": {"score": 3, "justification": "Clear wording, with minor ambiguity."},
                    "Verifiable": {"score": 4, "justification": "Can be easily tested or measured."}
                }
            },
            "notes": [
                "Each criterion must be scored from 0 to 4.",
                "Include a brief justification for each score.",
                "Follow the JSON format exactly as shown in output_format."
            ]
        },
        "requirement_to_evaluate": {
            "RequirementID": req_id,
            "RequirementText": requirement_text
        }
    }

    return [
        {
            "role": "user",
            "content": f"Please evaluate this requirement using the following structured JSON prompt:\n\n{json.dumps(prompt, indent=2)}"
        }
    ]



def evaluate_requirement_quality_iso(requirement_text, req_id):
    try:
        prompt = get_updated_iso_prompt(requirement_text, req_id)
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=prompt,
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
            row[f"{dim}_Score"] = val.get("score", "N/A")
            row[f"{dim}_Justification"] = val.get("justification", "N/A")
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


def show_type_distribution_chart(requirements):
    labels = ['Functional', 'Non-Functional', 'Other']
    counts = [0, 0, 0]

    for req in requirements:
        if req["RequirementID"].startswith("FR"):
            counts[0] += 1
        elif req["RequirementID"].startswith("NFR"):
            counts[1] += 1
        else:
            counts[2] += 1

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        wedgeprops=dict(width=0.4, edgecolor='white'),
        textprops=dict(color="black", fontsize=10)
    )
    ax.set_title(" Requirement Type Distribution", fontsize=12)
    plt.setp(autotexts, weight="bold", fontsize=10)

    with st.expander("📊 View Requirement Type Pie Chart", expanded=False):
        st.pyplot(fig)


def show_quality_score_chart(requirements):
    score_map = defaultdict(list)

    for req in requirements:
        qa = req.get("quality_evaluation", {})
        for criterion, result in qa.items():
            try:
                score = float(result.get("score", 0))
                score_map[criterion].append(score)
            except ValueError:
                continue

    avg_scores = {k: sum(v) / len(v) if v else 0 for k, v in score_map.items()}
    criteria = list(avg_scores.keys())
    scores = [avg_scores[c] for c in criteria]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(criteria, scores)
    ax.set_title("Average Quality Scores per ISO 29148 Criterion", fontsize=12)
    ax.set_ylabel("Average Score (1 to 5)")
    ax.set_ylim(0, 5)
    plt.xticks(rotation=30, ha="right")

    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.annotate(f"{score:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    with st.expander("📊 View Average Quality Scores Bar Chart", expanded=False):
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

    # Detect actual document language
    try:
        detected_lang_code = detect(extracted_text)
        detected_lang_name = [lang for lang, code in language_code_map.items() if code == detected_lang_code]
        detected_lang_name = detected_lang_name[0] if detected_lang_name else detected_lang_code
    except Exception:
        detected_lang_code = "unknown"
        detected_lang_name = "Unknown"

    # Warn if mismatch
    if detected_lang_code != source_lang_code:
        st.warning(
            f"⚠️ The uploaded document seems to be in **{detected_lang_name}**, "
            f"but you selected **{selected_language}**. Please verify the language setting."
        )

    # Proceed with translation
    extracted_text = translate_to_english(extracted_text, source_lang_code)

    # ⛑️ Initialize session state if not already
    if "all_reqs" not in st.session_state:
        st.session_state["all_reqs"] = []
        st.session_state["final_response"] = {
            "functional_requirements": [],
            "non_functional_requirements": [],
            "missing_critical_requirements": [],
            "recommendations": []
        }

    if st.button("Extract & Evaluate Requirements"):
        if not extracted_text:
            st.warning("⚠️ No text found to analyze.")
        else:
            st.info("📤 Sending document to OpenAI for requirement extraction...")

            chunks = split_text(extracted_text)
            all_results = []

            objectives_summary = []
            for idx, chunk in enumerate(chunks):
                try:
                    objective_prompt = get_document_objectives_prompt(chunk)
                    response = client.chat.completions.create(
                        model="gpt-4-turbo",
                        messages=objective_prompt
                    )
                    objectives_summary.append(response.choices[0].message.content.strip())
                except Exception as e:
                    st.warning(f"⚠️ Objective extraction failed on chunk {idx + 1}: {str(e)}")
                    objectives_summary.append("")

            # Save to session
            st.session_state["document_objectives"] = "\n\n".join([o for o in objectives_summary if o])

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

            cleaned_text = clean_text(extracted_text)

            with st.expander("🖍️ View Full Document", expanded=False):
                st.markdown(f"<div style='white-space: pre-wrap; font-size: 15px;'>{cleaned_text}</div>", unsafe_allow_html=True)

            # ✅ Persist final response and requirements
            st.session_state["final_response"] = final_response
            st.session_state["all_reqs"] = final_response["functional_requirements"] + final_response["non_functional_requirements"]

    # ✅ Show charts and requirements if already available in state
    if st.session_state["all_reqs"]:
        all_reqs = st.session_state["all_reqs"]
        show_summary_metrics(all_reqs)
        show_type_distribution_chart(all_reqs)


        st.subheader("📋 Extracted Requirements")

        if "document_objectives" in st.session_state:
            with st.expander("🎯 Document Objectives, Goals, and Purpose", expanded=False):
                st.markdown(f"<div style='white-space: pre-wrap; font-size: 15px;'>{st.session_state['document_objectives']}</div>", unsafe_allow_html=True)

        st.markdown("### 🛠 Functional Requirements")

        for req in sorted([r for r in all_reqs if r["RequirementID"].startswith("FR")], key=lambda r: r["RequirementID"]):
            with st.expander(f"**{req['RequirementID']}**: {req['Requirement']}", expanded=False):
                state_key = f"{req['RequirementID']}_evaluated"
                button_key = f"{req['RequirementID']}_evaluate_btn"
                if state_key in st.session_state:
                    req["quality_evaluation"] = st.session_state[state_key]
                    for dim, val in req["quality_evaluation"].items():
                        score = val.get("score", "N/A")
                        justification = val.get("justification", "No reason provided.")
                        st.markdown(f"• **{dim}**: Score **{score}** — _{justification}_")
                elif st.button(f"🧪 Evaluate {req['RequirementID']}", key=button_key):
                    with st.spinner("Evaluating..."):
                        result = evaluate_requirement_quality_iso(req["Requirement"], req["RequirementID"])
                        req["quality_evaluation"] = result.get("ISO29148_QualityAssessment", {})
                        st.session_state[state_key] = req["quality_evaluation"]
                    for dim, val in req["quality_evaluation"].items():
                        score = val.get("score", "N/A")
                        justification = val.get("justification", "No reason provided.")
                        st.markdown(f"• **{dim}**: Score **{score}** — _{justification}_")

        st.markdown("### 🎯 Non-Functional Requirements")
        for req in sorted([r for r in all_reqs if r["RequirementID"].startswith("NFR")], key=lambda r: r["RequirementID"]):
            with st.expander(f"**{req['RequirementID']}**: {req['Requirement']}", expanded=False):
                state_key = f"{req['RequirementID']}_evaluated"
                button_key = f"{req['RequirementID']}_evaluate_btn"
                if state_key in st.session_state:
                    req["quality_evaluation"] = st.session_state[state_key]
                    for dim, val in req["quality_evaluation"].items():
                        score = val.get("score", "N/A")
                        justification = val.get("justification", "No reason provided.")
                        st.markdown(f"• **{dim}**: Score **{score}** — _{justification}_")
                elif st.button(f"🧪 Evaluate {req['RequirementID']}", key=button_key):
                    with st.spinner("Evaluating..."):
                        result = evaluate_requirement_quality_iso(req["Requirement"], req["RequirementID"])
                        req["quality_evaluation"] = result.get("ISO29148_QualityAssessment", {})
                        st.session_state[state_key] = req["quality_evaluation"]
                    for dim, val in req["quality_evaluation"].items():
                        score = val.get("score", "N/A")
                        justification = val.get("justification", "No reason provided.")
                        st.markdown(f"• **{dim}**: Score **{score}** — _{justification}_")


if "all_reqs" in st.session_state:
    all_reqs = st.session_state["all_reqs"]

    if "json_report" not in st.session_state:
        st.session_state["json_report"] = json.dumps(st.session_state["final_response"], indent=4)

    if "excel_buffer" not in st.session_state:
        excel_df = export_to_excel(all_reqs)
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            excel_df.to_excel(writer, index=False)
        buffer.seek(0)
        st.session_state["excel_buffer"] = buffer

    if "csv_report" not in st.session_state:
        st.session_state["csv_report"] = export_to_excel(all_reqs).to_csv(index=False)

# ✅ Show extra sections only if requirements exist
if "all_reqs" in st.session_state and st.session_state["all_reqs"]:

    # 🔁 Manage Evaluation State
    st.markdown("### 🧹 Manage Evaluation State")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔁 Reset All Evaluations"):
            keys_to_remove = [key for key in st.session_state if key.endswith("_evaluated")]
            for key in keys_to_remove:
                del st.session_state[key]
            st.success("✅ All requirement evaluations have been reset.")
            st.rerun()

    with col2:
        if st.button("🧪 Evaluate All Requirements"):
            progress_placeholder = st.empty()
            spinner = st.empty()

            with spinner:
                with st.spinner("Evaluating all requirements..."):
                    total = len(st.session_state["all_reqs"])
                    for idx, req in enumerate(st.session_state["all_reqs"]):
                        req_id = req["RequirementID"]
                        state_key = f"{req_id}_evaluated"
                        if state_key not in st.session_state:
                            result = evaluate_requirement_quality_iso(req["Requirement"], req_id)
                            req["quality_evaluation"] = result.get("ISO29148_QualityAssessment", {})
                            st.session_state[state_key] = req["quality_evaluation"]

                        progress_placeholder.markdown(
                            f"🔍 Evaluated **{req_id}** ({idx + 1}/{total})..."
                        )
            spinner.empty()
            progress_placeholder.success("✅ All requirements evaluated!")
            st.session_state["all_evaluated"] = True
            st.rerun()

    if st.session_state.get("all_evaluated"):
        show_quality_score_chart(st.session_state["all_reqs"])
        del st.session_state["all_evaluated"]

    # 📥 Download section
    st.markdown("### ⬇️ Download Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        if "json_report" in st.session_state:
            st.download_button(
                "📥 Download JSON Report",
                st.session_state["json_report"],
                "requirement_analysis.json",
                "application/json"
            )

    with col2:
        if "excel_buffer" in st.session_state:
            st.download_button(
                "📊 Download Excel Report",
                st.session_state["excel_buffer"],
                "requirement_analysis.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    with col3:
        if "csv_report" in st.session_state:
            st.download_button(
                "📄 Download CSV Report",
                st.session_state["csv_report"],
                "requirement_analysis.csv",
                "text/csv"
            )

    # 🗣️ User Feedback
    st.markdown("### 🗣️ User Feedback")

    if "feedback_log" not in st.session_state:
        st.session_state["feedback_log"] = []

    user_feedback = st.text_area("💬 Leave your feedback about this analysis:")

    if st.button("✅ Submit Feedback"):
        feedback_entry = {
            "filename": uploaded_file.name if uploaded_file else "unknown_file",
            "feedback": user_feedback,
            "total_requirements": len(st.session_state.get("all_reqs", []))
        }
        st.session_state["feedback_log"].append(feedback_entry)
        st.success("✅ Thank you for your feedback!")

        # Save to local file
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state["feedback_log"], f, indent=2)