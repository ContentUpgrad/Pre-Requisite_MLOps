'''
constants.py contains all variables
where constant values are assigned
'''

# Latest model to be used in the inference process
MODEL_NAME = 'aditya_model1_adaboost.joblib'
# Column names of the source tabular data
ORIGINAL_FEATURES = ['age', 'sex', 'chest_pain_type', 'resting_bp', 'cholestoral',
       'fasting_blood_sugar', 'restecg', 'max_hr', 'exang', 'oldpeak', 'slope',
       'num_major_vessels', 'thal']
# Categorical features that needs to be one hot encoded
FEATURES_TO_ENCODE = ['thal', 'slope', 'chest_pain_type', 'restecg']
# Feature names after one hot encoding
ONE_HOT_ENCODED_FEATURES = ['age', 'sex', 'resting_bp', 'cholestoral', 'fasting_blood_sugar',
       'max_hr', 'exang', 'oldpeak', 'num_major_vessels', 'thal_0', 'thal_1',
       'thal_2', 'thal_3', 'slope_0', 'slope_1', 'slope_2',
       'chest_pain_type_0', 'chest_pain_type_1', 'chest_pain_type_2',
       'chest_pain_type_3', 'restecg_0', 'restecg_1', 'restecg_2']
