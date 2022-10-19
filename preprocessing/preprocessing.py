"""
To ensure that all jupyter notebokks always use the same preprocessing, all preprocessing steps are implemented in this file.
"""
import os
import pandas as pd
from pandas import DataFrame
from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from imblearn.over_sampling import RandomOverSampler

columns_to_remove = ["FMONTH", "IDATE", "IDAY", "IMONTH", "IYEAR", "DISPCODE", "SEQNO", "_PSU", "CTELENUM", "EXACTOT1",
                     "EXACTOT2", "PCDMDECN"]
diabetes_columns = ["Yes", "Yes, but only during pregnancy", "No", "No, but pre-diabetes", "Don't know", "Refused"]


def load_dataset():
    return pd.read_csv("../brfss_dataset/2015.csv")


def get_preprocessed_brfss_dataset() -> Tuple[DataFrame, DataFrame]:
    dataset = load_dataset()
    return preprocess_brfss_dataset(dataset)


def get_preprocessed_brfss_train_test_split(oversampling=False) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    dataset = load_dataset()
    preprocessed_dataset, target = preprocess_brfss_dataset(dataset)
    if oversampling:
        preprocessed_dataset, target = oversample_dataset(preprocessed_dataset, target)
    return split_dataset(preprocessed_dataset, target)


def get_preprocessed_brfss_train_test_split_one_hot_encoded(oversampling=False) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    dataset = load_dataset()
    preprocessed_dataset, target = preprocess_brfss_dataset(dataset)
    if oversampling:
        preprocessed_dataset, target = oversample_dataset(preprocessed_dataset, target)
    one_hot_encoder = OneHotEncoder()
    target = pd.DataFrame(one_hot_encoder.fit_transform(target).toarray(), columns=diabetes_columns)
    return split_dataset(preprocessed_dataset, target)


def split_dataset(preprocessed_dataset, target) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    data_train, data_test, target_train, target_test = train_test_split(
        preprocessed_dataset, target, test_size=0.2, random_state=42, stratify=target)
    return data_train, data_test, target_train, target_test


def oversample_dataset(dataset, target) -> Tuple[DataFrame, DataFrame]:
    over_sampler = RandomOverSampler()
    brfss_balanced, brfss_balanced_target = over_sampler.fit_resample(dataset, target)
    return brfss_balanced, brfss_balanced_target


def preprocess_brfss_dataset(dataset: DataFrame) -> Tuple[DataFrame, DataFrame]:
    dataset = dataset.dropna(subset=['DIABETE3'])
    brfss_target = pd.DataFrame(dataset["DIABETE3"])
    brfss_preprocessed = dataset.drop(columns="DIABETE3")
    brfss_preprocessed = brfss_preprocessed.drop(columns=columns_to_remove)
    brfss_preprocessed = brfss_preprocessed.fillna(0)
    return brfss_preprocessed, brfss_target


def download_brfss_dataset(kaggle_username, kaggle_api_key):
    os.environ['KAGGLE_USERNAME'] = kaggle_username
    os.environ['KAGGLE_KEY'] = kaggle_api_key
    import kaggle
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('cdc/behavioral-risk-factor-surveillance-system', path='../brfss_dataset',
                                      unzip=True)
    pass
