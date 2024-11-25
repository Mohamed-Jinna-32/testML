import numpy as np
import pickle
import pandas as pd


with open('scaler', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)


columns = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 
           'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 
           'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 
           'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 
           'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 
           'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']

new_data = np.array([[15.5, 16.2, 120.5, 1000.0, 0.115, 0.230, 0.250, 0.160, 0.120,
                      0.127, 0.208, 132.0, 1030.0, 0.135, 0.260, 0.320, 0.250, 0.220,
                      0.190, 0.105, 0.460, 0.155, 0.205, 0.320, 0.480, 0.210, 0.210,
                      130.0, 118.0, 141.0]])


new_data_df = pd.DataFrame(new_data, columns=columns)


new_data_scaled = loaded_scaler.transform(new_data_df)


with open('final_logistic_regression_model', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

def predict(new_data):
    prediction = loaded_model.predict(new_data_scaled)
    print("Prediction:", prediction)

    y_pred = loaded_model.predict_proba(new_data_scaled)[0, 1]
    return y_pred


predict(new_data)
