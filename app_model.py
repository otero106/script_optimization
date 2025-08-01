# app.py
import streamlit as st
from emotion_analysis_page import emotion_page
from retention_analysis_page import retention_page
from upload_page import upload_page
from character_analysis import character_page
import re

st.set_page_config(page_title="Script Analyzer", layout="wide")

# Define the pages
pages = [
    st.Page(upload_page, title="Home", icon="📁"),
    st.Page(emotion_page, title="Emotion Analysis", icon="🎭"),
    st.Page(character_page, title="Character Analysis", icon="👥"),
    st.Page(retention_page, title="Retention Analysis", icon="📈"),
]

# Create the navigation
nav = st.navigation(pages)

# Run the selected page
nav.run()
