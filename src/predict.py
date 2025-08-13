import pickle
import pandas as pd

with open("models/iris_model.pkl", "rb") as f:
    model = pickle.load(f)

def predict(features: list):
    df = pd.DataFrame([features], columns=[
        "sepal length (cm)", "sepal width (cm)",
        "petal length (cm)", "petal width (cm)"
    ])
    pred = model.predict(df)[0]
    return pred
