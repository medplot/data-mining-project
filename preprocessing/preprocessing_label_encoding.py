from pandas import DataFrame
from typing import Tuple

from preprocessing import get_preprocessed_brfss_dataset, oversample_dataset, undersample_dataset


# Returns preprocessed dataset where all ordinal values keep their label encoding
# This function does not include any sampling
def get_preprocessed_brfss_dataset_label_encoded() -> Tuple[DataFrame, DataFrame]:
    return get_preprocessed_brfss_dataset()


# Returns preprocessed dataset where all ordinal values keep their label encoding
# This function includes oversampling
def get_preprocessed_brfss_dataset_label_encoded_oversampled() -> Tuple[DataFrame, DataFrame]:
    dataset, target = get_preprocessed_brfss_dataset()
    dataset, target = oversample_dataset(dataset, target)
    return dataset, target


# Returns preprocessed dataset where all ordinal values keep their label encoding
# This function includes undersampling
def get_preprocessed_brfss_dataset_label_encoded_undersampled() -> Tuple[DataFrame, DataFrame]:
    dataset, target = get_preprocessed_brfss_dataset()
    dataset, target = undersample_dataset(dataset, target)
    return dataset, target
