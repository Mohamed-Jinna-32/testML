import streamlit as st
import pickle
import numpy as np

with open('final_logistic_regression_model', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


def predict(features):
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)[0, 1]
    return prediction[0], probability


st.title("Breast Cancer Prediction using Logistic Regression")
st.write("Enter the feature values below:")


radius_mean = st.number_input("Radius Mean", min_value=0.0, value=15.5)
texture_mean = st.number_input("Texture Mean", min_value=0.0, value=16.2)
perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, value=120.5)
area_mean = st.number_input("Area Mean", min_value=0.0, value=1000.0)
smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, value=0.115)
compactness_mean = st.number_input("Compactness Mean", min_value=0.0, value=0.230)
concavity_mean = st.number_input("Concavity Mean", min_value=0.0, value=0.250)
concave_points_mean = st.number_input("Concave Points Mean", min_value=0.0, value=0.160)
symmetry_mean = st.number_input("Symmetry Mean", min_value=0.0, value=0.120)
fractal_dimension_mean = st.number_input("Fractal Dimension Mean", min_value=0.0, value=0.127)

radius_se = st.number_input("Radius SE", min_value=0.0, value=0.208)
texture_se = st.number_input("Texture SE", min_value=0.0, value=132.0)
perimeter_se = st.number_input("Perimeter SE", min_value=0.0, value=1030.0)
area_se = st.number_input("Area SE", min_value=0.0, value=0.135)
smoothness_se = st.number_input("Smoothness SE", min_value=0.0, value=0.260)
compactness_se = st.number_input("Compactness SE", min_value=0.0, value=0.320)
concavity_se = st.number_input("Concavity SE", min_value=0.0, value=0.250)
concave_points_se = st.number_input("Concave Points SE", min_value=0.0, value=0.220)
symmetry_se = st.number_input("Symmetry SE", min_value=0.0, value=0.190)
fractal_dimension_se = st.number_input("Fractal Dimension SE", min_value=0.0, value=0.105)

radius_worst = st.number_input("Radius Worst", min_value=0.0, value=0.460)
texture_worst = st.number_input("Texture Worst", min_value=0.0, value=0.155)
perimeter_worst = st.number_input("Perimeter Worst", min_value=0.0, value=0.205)
area_worst = st.number_input("Area Worst", min_value=0.0, value=0.320)
smoothness_worst = st.number_input("Smoothness Worst", min_value=0.0, value=0.480)
compactness_worst = st.number_input("Compactness Worst", min_value=0.0, value=0.210)
concavity_worst = st.number_input("Concavity Worst", min_value=0.0, value=0.210)
concave_points_worst = st.number_input("Concave Points Worst", min_value=0.0, value=130.0)
symmetry_worst = st.number_input("Symmetry Worst", min_value=0.0, value=118.0)
fractal_dimension_worst = st.number_input("Fractal Dimension Worst", min_value=0.0, value=141.0)


features = [
    radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, 
    concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, 
    radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, 
    concavity_se, concave_points_se, symmetry_se, fractal_dimension_se, 
    radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, 
    compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst
]

if st.button('Predict'):
    prediction, probability = predict(features)
    st.write(f"Prediction: {'Malignant' if prediction == 1 else 'Benign'}")
    st.write(f"Probability: {probability:.2f}")
