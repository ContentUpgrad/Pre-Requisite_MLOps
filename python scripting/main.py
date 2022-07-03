#!/usr/bin/env python
# coding: utf-8

# ## Heart Disease Inference
# In this script, we will try to look at
# the inference part of the heart disease classification solution

# ### Import Modules
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score
from utils import *
from constants import *
import argparse


# main function - starting point of the code
def main(model_name):
    '''
    main function - starting point of the code
    input: model name taken as input 
    (default: adaboost model)
    output: no return value, only prints outcome
    '''
    print("Starting execution of the inference code...")
    # in real-time use cases, this code should be replaced with live flowing data
    # use get_inference_data() from utils.py to fetch inference data
    inference_data, labels = get_inference_data()
    # use apply_pre_processing() from utils.py 
    # to apply necessary preprocessing as applied for training data
    processed_inference_data = apply_pre_processing(inference_data)
    # ### Load Saved Model
    model = joblib.load(model_name)
    # ### Prediction on inference data
    model.predict(processed_inference_data)
    # ### Scoring check on prediction
    print("Checking inference accuracy:")
    print(accuracy_score(labels, model.predict(processed_inference_data)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Running inference pipeline')
    parser.add_argument('--model',
                        default='adaboost',
                        help='select algorithm: svm or adaboost')
    args = parser.parse_args()
    print(f"Selected algorithm: {args.model}")
    if(args.model == 'svm'):
        model_name = 'aditya_model2_svm.joblib'
    else:
        model_name = 'aditya_model1_adaboost.joblib'
    main(model_name)
