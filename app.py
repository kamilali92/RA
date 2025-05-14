import streamlit as st
from data_loader import load_data, preprocess
from model import train_model
from predict import load_model, make_prediction
from utils import get_user_input

st.set_page_config(page_title="Crop Recommendation App 🌱", layout="wide")
st.title("🌿 Crop Recommendation using Machine Learning")

menu = ["Train Model", "Predict"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Train Model":
    st.subheader("Upload Dataset")
    uploaded_file = st.file_uploader("Upload your dataset (.csv)", type=["csv"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        df = preprocess(df)

        st.write("📊 Dataset Preview", df.head())
        st.write("✅ Columns:", df.columns.tolist())
        st.write("📏 Shape:", df.shape)

        target_column = 'label'  # For this dataset, label is the target

        if st.button("Train Model"):
            model, acc = train_model(df, target_column)
            st.success(f"✅ Model trained successfully! Accuracy: {acc:.2f}")

elif choice == "Predict":
    st.subheader("🌾 Enter Crop Conditions")
    model = load_model()

    user_input = get_user_input()  # From utils.py

    for feature in user_input:
        user_input[feature] = st.number_input(f"{feature}", value=float(user_input[feature]))

    if st.button("Predict Crop"):
        result = make_prediction(model, user_input)
        st.success(f"🌱 Recommended Crop: **{result}**")
