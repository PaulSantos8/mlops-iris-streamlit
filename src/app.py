import streamlit as st
from predict import predict
from sklearn.datasets import load_iris

iris = load_iris()
st.title("Clasificación Iris 🌸")
st.write("Introduce las medidas de la flor:")

sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.5)
sepal_width  = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 4.0)
petal_width  = st.slider("Petal width (cm)", 0.1, 2.5, 1.0)

if st.button("Predecir"):
    pred = predict([sepal_length, sepal_width, petal_length, petal_width])
    st.success(f"La predicción es: {iris.target_names[pred]}")
