import joblib
import numpy as np

model = joblib.load("models/iris_model.pkl")

def predict_species(features):
    prediction = model.predict([features])
    return int(prediction[0])