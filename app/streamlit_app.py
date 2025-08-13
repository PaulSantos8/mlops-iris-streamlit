import streamlit as st
import numpy as np
from src.model.predict_model import predict_species
from src.data.load_data import get_iris_data

st.title("Iris Flower Classifier")

df, target_names = get_iris_data()

sepal_length = st.slider('Sepal length (cm)', float(df.iloc[:,0].min()), float(df.iloc[:,0].max()))
sepal_width = st.slider('Sepal width (cm)', float(df.iloc[:,1].min()), float(df.iloc[:,1].max()))
petal_length = st.slider('Petal length (cm)', float(df.iloc[:,2].min()), float(df.iloc[:,2].max()))
petal_width = st.slider('Petal width (cm)', float(df.iloc[:,3].min()), float(df.iloc[:,3].max()))

if st.button('Predict'):
    features = [sepal_length, sepal_width, petal_length, petal_width]
    result = predict_species(features)
    st.success(f"Predicted Species: {target_names[result]}")