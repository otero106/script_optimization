# 🎬 Script Analyzer

Welcome to the **Script Analyzer**, a Streamlit-based web app for analyzing film or TV scripts using natural language processing. This tool extracts key emotional and structural insights from PDF or CSV scripts, helping storytellers and researchers better understand dialogue dynamics, character emotions, and scene structure.

---

## 🚀 Features

- 🔍 **Automatic Script Parsing** from raw PDF files
- 🧠 **Emotion Detection** using a DistilBERT-based model
- 💬 **Sentiment Analysis** with TextBlob
- 📊 **Visualizations** including:
  - Emotion heatmap
  - Scene-level emotion breakdown
  - Character-centric emotion maps
  - Emotion arcs over time
- 📈 **Dialogue Metrics** like cumulative word count, lead character usage, and visual density

---

## 📁 Supported File Types

- `.pdf` – Raw scripts (line-by-line text)
- `.csv` – Pre-annotated scripts with key columns already available

---

## 🛠️ Installation

Make sure you have Python 3.8+ and `pip` installed. Then:

```bash
# 1. Clone this repo or navigate to your script folder
cd /path/to/your/folder

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source . venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
