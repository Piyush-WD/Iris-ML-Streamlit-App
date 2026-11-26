import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Iris Species Predictor", page_icon="🌸", layout="centered")

MODEL_PATH = Path(__file__).parent / "model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("🌸 Iris Species Predictor")
st.write(
    "Explore a machine-learning classification model. Adjust flower measurements "
    "and predict whether an iris is Setosa, Versicolor, or Virginica."
)

st.info("Educational demonstration using the classic Iris dataset. Do not use for real-world decisions.")

try:
    model = load_model()
except FileNotFoundError:
    st.error("The trained model file (model.joblib) is missing. Run `python train_model.py` first.")
    st.stop()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

st.subheader("Enter flower measurements")

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.1, 0.1)
    sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.5, 0.1)
with col2:
    petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 1.4, 0.1)
    petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 0.2, 0.1)

input_data = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
)

if st.button("Predict iris species", type="primary", use_container_width=True):
    try:
        probabilities = model.predict_proba(input_data)[0]
        predicted_index = int(probabilities.argmax())
        predicted_species = model.classes_[predicted_index]

        st.success(f"Predicted species: **{predicted_species}**")
        st.metric("Model confidence", f"{probabilities[predicted_index]:.1%}")

        results = pd.DataFrame({
            "Species": model.classes_,
            "Probability": probabilities
        }).sort_values("Probability", ascending=False)

        st.subheader("Prediction probabilities")
        st.bar_chart(results.set_index("Species"))
        st.dataframe(
            results.assign(Probability=results["Probability"].map(lambda x: f"{x:.1%}")),
            hide_index=True,
            use_container_width=True
        )
    except Exception as exc:
        st.error(f"Prediction could not be generated. Please check the inputs. Details: {exc}")

with st.expander("About this project"):
    st.write(
        "This app uses a Random Forest classifier trained on the built-in Iris dataset "
        "from scikit-learn. The model uses four measurements: sepal length, sepal width, "
        "petal length, and petal width. The app loads a saved model with joblib and "
        "displays class probabilities."
    )
