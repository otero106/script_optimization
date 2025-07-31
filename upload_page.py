# upload_page.py
import streamlit as st
import pandas as pd
from utils import process_pdf_into_dataframe, add_bg_from_url

def upload_page():
    add_bg_from_url()
    
    st.image("https://cdn.prod.website-files.com/64b83e9317dc3622290fd4fa/65a9178afbb21bd44cbf074f_toonstar-logo-removebg-preview.png", width=300)
    st.title("🎬 Script Analyzer")
    st.markdown("Upload your annotated script **CSV** or raw **PDF** script file to begin analysis.")

    uploaded_file = st.file_uploader("📁 Upload a script file", type=["csv", "pdf"])

    if uploaded_file:
        if 'df' not in st.session_state or st.session_state.get('file_name') != uploaded_file.name:
            with st.spinner("Processing file..."):
                if uploaded_file.name.endswith(".pdf"):
                    st.session_state.df = process_pdf_into_dataframe(uploaded_file)
                elif uploaded_file.name.endswith(".csv"):
                    st.session_state.df = pd.read_csv(uploaded_file)
                st.session_state.file_name = uploaded_file.name
            st.success("✅ File processed successfully! Navigate to the analysis pages.")
        
    if 'df' in st.session_state:
        st.markdown("---")
        st.subheader("Data Preview")
        st.dataframe(st.session_state.df.head())