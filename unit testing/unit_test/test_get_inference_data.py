from main import *
import pandas as pd


class TestGetInferenceData:
    def test_method_output_type(self):
        df = pd.DataFrame()
        inference_data, labels = get_inference_data()
        assert type(inference_data)==type(df), "inference data should be a dataframe"

    def test_method_output_len(self):
        df = pd.DataFrame()
        inference_data, labels = get_inference_data()
        assert len(labels)==len(inference_data), "length of data and labels are not matching"