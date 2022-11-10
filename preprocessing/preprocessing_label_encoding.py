from typing import Tuple

from pandas import DataFrame

from preprocessing.preprocessing import get_preprocessed_brfss_dataset, oversample_dataset, undersample_dataset, \
    get_train_test_split


# Returns preprocessed dataset where all ordinal values keep their label encoding.
# This function does not include any sampling.
def get_preprocessed_brfss_dataset_label_encoded() -> Tuple[DataFrame, DataFrame]:
    dataset, target = get_preprocessed_brfss_dataset()
    return dataset, target


# Returns a train/test split of the preprocessed dataset where all ordinal values keep their label encoding.
# This function does not include any sampling.
def get_preprocessed_brfss_dataset_label_encoded_train_test_split() \
        -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    dataset, target = get_preprocessed_brfss_dataset_label_encoded()
    data_train, data_test, target_train, target_test = get_train_test_split(dataset, target)
    return data_train, data_test, target_train, target_test


# Returns a train/test split of the preprocessed dataset where all ordinal values keep their label encoding.
# This function includes oversampling
def get_preprocessed_brfss_dataset_label_encoded_train_test_split_oversampled() \
        -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    data_train, data_test, target_train, target_test = get_preprocessed_brfss_dataset_label_encoded_train_test_split()
    data_train, target_train = oversample_dataset(data_train, target_train)
    return data_train, data_test, target_train, target_test


# Returns a train/test split of the preprocessed dataset where all ordinal values keep their label encoding.
# This function includes oversampling
def get_preprocessed_brfss_dataset_label_encoded_train_test_split_undersampled() \
        -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    data_train, data_test, target_train, target_test = get_preprocessed_brfss_dataset_label_encoded_train_test_split()
    data_train, target_train = undersample_dataset(data_train, target_train)
    return data_train, data_test, target_train, target_test
