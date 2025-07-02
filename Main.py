import streamlit as st
from streamlit_option_menu import option_menu

# Set Streamlit page config
st.set_page_config(page_title="LLM-Powered Requirements Tool", layout="wide")

# Custom CSS for full page styling
custom_css = """
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

        /* Global Styling */
        html, body, [class*="stApp"] {
            background-color: #121212;
            color: #EAEAEA;
            font-family: 'Poppins', sans-serif;
        }

        /* Change the Streamlit block container (main content) */
        .block-container {
            padding-top: 2rem;
            max-width: 85%;
            margin: auto;
        }

        /* Custom Navbar Styling */
        div[data-testid="stHorizontalBlock"] {
            background-color: #1E1E1E;
            padding: 14px;
            border-radius: 10px;
            box-shadow: 0px 4px 12px rgba(255, 255, 255, 0.08);
            display: flex;
            justify-content: center;
            gap: 15px;
        }

        /* Navbar button styling */
        div[data-testid="stHorizontalBlock"] button {
            background-color: #2C2C2C;
            color: white;
            border-radius: 8px;
            padding: 12px 20px;
            transition: all 0.3s ease-in-out;
            font-size: 16px;
            font-weight: 500;
            border: none;
        }

        /* Navbar hover effect */
        div[data-testid="stHorizontalBlock"] button:hover {
            background-color: #4CAF50;
            color: white;
        }

        /* Highlight the selected tab */
        div[data-testid="stHorizontalBlock"] button[aria-pressed="true"] {
            background-color: #E63946 !important;
            color: white !important;
            font-weight: 600 !important;
        }

        /* Customize headings */
        h1, h2, h3 {
            font-weight: 600;
        }
        
        /* Improve list appearance */
        ul {
            list-style-type: "✅ ";
            padding-left: 20px;
        }

        /* Custom Button Styling */
        .stButton>button {
            background-color: #ff4b4b;
            color: white;
            font-size: 18px;
            border-radius: 10px;
            padding: 12px;
            transition: 0.3s ease-in-out;
            border: none;
        }

        /* Button Hover Effect */
        .stButton>button:hover {
            background-color: #E63946;
            transform: scale(1.05);
        }

        /* Fix Page Spacing */
        .main-content {
            max-width: 90%;
            margin: auto;
        }

    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.title("🚀 Welcome to the LLM-Powered Requirements Extraction Tool")
st.markdown("""
            This tool allows you to:
        
        - 📂 Upload project documents (**PDF, DOCX, TXT**)
        
        - ✅ Extract & categorize **Functional & Non-Functional Requirements**
        
        - 📊 Validate requirement quality based on **ISO 29148 Standards**
        
        - 🤖 Receive AI-suggested **requirement improvements**
        
        - 📝 **Edit and update** requirements in real-time
        
        - 📤 **Export** final requirements in **DOCX, CSV, or PDF**
    """)

