from typing import Tuple, Union

from pandas import DataFrame

from preprocessing.preprocessing import get_preprocessed_brfss_dataset, oversample_dataset, undersample_dataset, \
    get_train_validation_test_split


# Returns preprocessed dataset where all ordinal values keep their label encoding.
# This function does not include any sampling.
def get_preprocessed_brfss_dataset_label_encoded() -> Tuple[DataFrame, DataFrame]:
    return get_preprocessed_brfss_dataset()


# Returns a train/validation split of the preprocessed dataset where all ordinal values keep their label encoding.
# If the parameter include_test_data is set to true, test data will also be returned.
# This function does not include any sampling.
def get_preprocessed_brfss_dataset_label_encoded_train_test_split(include_test_data=False) \
        -> Union[Tuple[DataFrame, DataFrame, DataFrame, DataFrame], Tuple[
            DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]]:
    dataset, target = get_preprocessed_brfss_dataset_label_encoded()
    return get_train_validation_test_split(dataset, target, include_test_data)


# Returns a train/validation split of the preprocessed dataset where all ordinal values keep their label encoding.
# If the parameter include_test_data is set to true, test data will also be returned.
# This function includes oversampling
def get_preprocessed_brfss_dataset_label_encoded_train_test_split_oversampled(include_test_data=False) \
        -> Union[Tuple[DataFrame, DataFrame, DataFrame, DataFrame], Tuple[
            DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]]:
    if include_test_data:
        data_train, data_validation, data_test, target_train, target_validation, target_test = \
            get_preprocessed_brfss_dataset_label_encoded_train_test_split(include_test_data)
        data_train, target_train = oversample_dataset(data_train, target_train)
        return data_train, data_validation, data_test, target_train, target_validation, target_test
    else:
        data_train, data_validation, target_train, target_validation = \
            get_preprocessed_brfss_dataset_label_encoded_train_test_split(include_test_data)
        data_train, target_train = oversample_dataset(data_train, target_train)
        return data_train, data_validation, target_train, target_validation


# Returns a train/validation split of the preprocessed dataset where all ordinal values keep their label encoding.
# If the parameter include_test_data is set to true, test data will also be returned.
# This function includes oversampling
def get_preprocessed_brfss_dataset_label_encoded_train_test_split_undersampled(include_test_data=False) \
        -> Union[Tuple[DataFrame, DataFrame, DataFrame, DataFrame], Tuple[
            DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]]:
    if include_test_data:
        data_train, data_validation, data_test, target_train, target_validation, target_test = \
            get_preprocessed_brfss_dataset_label_encoded_train_test_split(include_test_data)
        data_train, target_train = undersample_dataset(data_train, target_train)
        return data_train, data_validation, data_test, target_train, target_validation, target_test
    else:
        data_train, data_validation, target_train, target_validation = \
            get_preprocessed_brfss_dataset_label_encoded_train_test_split(include_test_data)
        data_train, target_train = undersample_dataset(data_train, target_train)
        return data_train, data_validation, target_train, target_validation
