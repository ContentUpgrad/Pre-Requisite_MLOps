import warnings
warnings.filterwarnings("ignore")
from utils import *
import pandas as pd


class TestDataNormalization:
    def test_norm_data_len(self):
        inference_data, labels = get_inference_data()
        norm_df = normalize_data(inference_data)
        assert len(inference_data)==len(norm_df),"Length has changed after normalization"

    def test_norm_data_valuerange(self):
        inference_data, labels = get_inference_data()
        norm_df = normalize_data(inference_data)
        assert int(norm_df.max().max()) <= 1,"Max value after normalization should be 1"
        assert int(norm_df.min().min()) >= 0,"Min value after normalization should be 0"