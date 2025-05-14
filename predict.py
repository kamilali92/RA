import joblib
import pandas as pd

def load_model():
    return joblib.load('model.pkl')

def make_prediction(model, input_dict):
    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]
    return prediction
