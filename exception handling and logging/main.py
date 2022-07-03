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
import logging


# main function - starting point of the code
def main(model_name, logger):
    '''
    main function - starting point of the code
    input: model name taken as input 
    (default: adaboost model)
    output: no return value, only prints outcome
    '''
    try:
        print("Starting execution of the inference code...")
        logger.info("Started execution. Fetching data now ...")
        # in real-time use cases, this code should be replaced with live flowing data
        # use get_inference_data() from utils.py to fetch inference data
        inference_data, labels = get_inference_data()
        logger.info("Data fetched. Applying pre-processing now ...")
        # use apply_pre_processing() from utils.py 
        # to apply necessary preprocessing as applied for training data
        processed_inference_data = apply_pre_processing(inference_data)
        logger.info("Pre-processing is completed. Loading trained model now ...")
        # ### Load Saved Model
        model = joblib.load(model_name)
        logger.info("Trained model is loaded. Executing trained model on inference data ...")
        # ### Prediction on inference data
        model.predict(processed_inference_data)
        # ### Scoring check on prediction
        print("Checking inference accuracy:")
        print(accuracy_score(labels, model.predict(processed_inference_data)))
        logger.info("Execution is complete.")
    except Exception as e:
        print("--------Error!!!--------")
        logger.error("Encountered error. Please check.")
        logger.error(e)
        print(e)


if __name__ == "__main__":
    # Create and configure logger
    logging.basicConfig(filename="inference_pipe_exec.log",
                    format='%(asctime)s %(message)s',
                    filemode='a')

    # Creating an object
    logger = logging.getLogger()

    # Setting the threshold of logger to DEBUG
    logger.setLevel(logging.DEBUG)

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
    main(model_name, logger)
