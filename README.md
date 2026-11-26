# Iris Species Predictor — Streamlit ML Web App

An interactive machine-learning web app that predicts an Iris flower species from four measurements and displays prediction probabilities.

## Features
- Interactive Streamlit sliders for four flower measurements
- Random Forest classification model
- Saved model loaded using `joblib`
- Cached model loading with `@st.cache_resource`
- Predicted species, confidence score, and class probabilities
- Friendly error handling for a missing model file

## Tech stack
Python, Streamlit, scikit-learn, pandas, joblib

## Run locally

```bash
python -m venv .venv
```

Activate the environment:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

Install dependencies and train the model:

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Model details
- Dataset: built-in Iris dataset from scikit-learn (150 samples, 4 numeric features, 3 classes)
- Algorithm: Random Forest Classifier (150 trees)
- Split: 80% training / 20% testing, stratified
- Evaluation: test accuracy and classification report are printed during training

## Deploy on Streamlit Community Cloud
1. Push this repository to GitHub, including `app.py`, `train_model.py`, `model.joblib`, and `requirements.txt`.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Select **Create app**, choose this repository and branch, and set the main file path to `app.py`.
4. Deploy and wait for the build to finish.
5. Open the public URL and test the sliders and prediction button.

## Project note
This is an educational demonstration using the classic Iris dataset. It is not intended for medical, agricultural, or other real-world decisions.
# Iris-ML-Streamlit-App
