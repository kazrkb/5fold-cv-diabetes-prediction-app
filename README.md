---
title: Diabetes Prediction
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Type-2 Diabetes Prediction App

A Streamlit web application utilizing 6 trained machine learning models to predict the risk of Type-2 Diabetes Mellitus based on clinical input parameters.

## How to Run Locally

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kazrkb/5fold-cv-diabetes-prediction-app.git
   cd 5fold-cv-diabetes-prediction-app
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the application**:
   ```bash
   streamlit run app.py
   ```
