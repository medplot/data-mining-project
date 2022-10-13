"""
To ensure that all jupyter notebokks always use the same preprocessing, all preprocessing steps are implemented in this file.
"""
import os
import pandas as pd
from pandas import DataFrame, Series
from typing import Tuple


def get_preprocessed_brfss_dataset() -> Tuple[DataFrame, Series]:
    data = pd.read_csv("../brfss_dataset/2015.csv")
    return preprocess_brfss_dataset(data)


def preprocess_brfss_dataset(dataset: DataFrame) -> Tuple[DataFrame, Series]:
    brfss_target = dataset["DIABETE3"]
    brfss_preprocessed = dataset.drop(columns="DIABETE3")
    return brfss_preprocessed, brfss_target


def download_brfss_dataset(kaggle_username, kaggle_api_key):
    os.environ['KAGGLE_USERNAME'] = kaggle_username
    os.environ['KAGGLE_KEY'] = kaggle_api_key
    import kaggle
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('cdc/behavioral-risk-factor-surveillance-system', path='../brfss_dataset',
                                      unzip=True)
    pass
