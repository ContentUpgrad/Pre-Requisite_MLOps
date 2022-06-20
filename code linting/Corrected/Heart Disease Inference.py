#!/usr/bin/env python
# coding: utf-8

# ## Heart Disease Classification
# In this script, we will try to look at
# the inference part of the heart disease classification solution

# ### Import Modules
import pandas as pd
import numpy as np
from sklearn import preprocessing
import joblib
from sklearn.metrics import accuracy_score


# ### Get Inference Data
# in real-time use cases, this method should be replaced with live flowing data
def get_inference_data():
    '''
    Method for loading inference data
    Example usage: inference_data, labels = get_inference_data()
    '''
    data = pd.read_csv("Data/heart.csv")
    data.drop_duplicates(subset=None, inplace=True)
    data.duplicated().any()
    df = data.sample(frac=1, random_state=2)
    # Taking last 20 records as an example only
    df = df.tail(20)
    return df[df.columns.drop('target')], df['target']


inference_data, labels = get_inference_data()

# ### Apply Same Pre-processing

# apply same pre-processing and feature engineering techniques
# as applied during the training process


def encode_features(df, features):
    '''
    Method for one-hot encoding all selected categorical fields
    Input: The method takes pandas dataframe and
    list of the feature names as input
    Output: Returns a dataframe with one-hot encoded features
    Example usage:
    one_hot_encoded_df = encode_features(dataframe, list_features_to_encode)
    '''
    # Implement these steps to prevent dimension mismatch during inference
    encoded_df = pd.DataFrame(columns=['age', 'sex', 'resting_bp',
                                       'cholestoral', 'fasting_blood_sugar',
                                       'max_hr', 'exang', 'oldpeak',
                                       'num_major_vessels', 'thal_0', 'thal_1',
                                       'thal_2', 'thal_3', 'slope_0',
                                       'slope_1', 'slope_2',
                                       'chest_pain_type_0',
                                       'chest_pain_type_1',
                                       'chest_pain_type_2',
                                       'chest_pain_type_3', 'restecg_0',
                                       'restecg_1', 'restecg_2'])
    placeholder_df = pd.DataFrame()

    # One-Hot Encoding using get_dummies for the specified categorical features
    for f in features:
        if(f in df.columns):
            encoded = pd.get_dummies(df[f])
            encoded = encoded.add_prefix(f + '_')
            placeholder_df = pd.concat([placeholder_df, encoded], axis=1)
        else:
            print('Feature not found')
            return df

    # Implement these steps to prevent dimension mismatch during inference
    for feature in encoded_df.columns:
        if feature in df.columns:
            encoded_df[feature] = df[feature]
        if feature in placeholder_df.columns:
            encoded_df[feature] = placeholder_df[feature]
    # fill all null values
    encoded_df.fillna(0, inplace=True)

    return encoded_df


def normalize_data(df):
    '''
    Normalize data using Min-Max Scaler
    Input: The method takes pandas dataframe as input
    Output: Returns a dataframe with normalized features
    Example usage:
    normalized_df = normalize_data(df)
    '''
    val = df.values
    min_max_normalizer = preprocessing.MinMaxScaler()
    norm_val = min_max_normalizer.fit_transform(val)
    df2 = pd.DataFrame(norm_val)
    return df2


def apply_pre_processing(data):
    '''
    Normalize data using Min-Max Scaler
    Input: The method takes pandas dataframe as input
    Output: Returns a dataframe with normalized features
    Example usage:
    normalized_df = normalize_data(df)
    '''
    features_to_encode = ['thal', 'slope', 'chest_pain_type', 'restecg']
    encoded = encode_features(data, features_to_encode)
    processed_data = normalize_data(encoded)
    # Please note this is fabricated inference data,
    # so just taking a small sample size
    return processed_data


processed_inference_data = apply_pre_processing(inference_data)


# ### Load Saved Model
model = joblib.load('aditya_model1_adaboost.joblib')


# ### Prediction on inference data
model.predict(processed_inference_data)


# ### Scoring check on prediction
accuracy_score(labels[-20:], model.predict(processed_inference_data))
