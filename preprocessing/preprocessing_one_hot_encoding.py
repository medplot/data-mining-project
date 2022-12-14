import pandas as pd
from pandas import DataFrame
from typing import Tuple, Union

from sklearn.preprocessing import OneHotEncoder

from preprocessing.preprocessing import get_preprocessed_brfss_dataset, diabetes_columns, oversample_dataset, \
    undersample_dataset, get_train_validation_test_split

# Excludes yes/no columns
ordinal_columns = ["GenHealth", "Checkup", "HighBP", "Income", "Age", "BMI", "Education", "Smoking", "PhysActivity"]

# Includes yes/no columns
all_ordinal_columns = ["GenHealth", "Healthcare", "MedCost", "Checkup", "HighBP", "HighChol", "HeartAttack", "AngiCoro",
                       "Stroke", "Asthma", "Arthritis", "Kidney", "Sex", "Income", "SodiumSalt", "Age", "BMI",
                       "Education", "Alcohol", "Smoking", "FruitCons", "VegetCons", "PhysActivity", "Muscles"]


# Returns preprocessed dataset where all columns with ordinal values that are not simply yes/no are one hot encoded.
# If the parameter is set to true, the target column will also be one hot encoded.
# This function does not include sampling.
def get_preprocessed_brfss_dataset_one_hot_encoded(target_one_hot_encoded=False) -> Tuple[DataFrame, DataFrame]:
    dataset, target = get_preprocessed_brfss_dataset()
    one_hot_encoder = OneHotEncoder()
    encoded = pd.DataFrame(
        one_hot_encoder.fit_transform(dataset[ordinal_columns]).toarray(),
        columns=one_hot_encoder.get_feature_names_out())
    dataset = dataset.join(encoded)
    dataset = dataset.drop(columns=ordinal_columns)
    if target_one_hot_encoded:
        target = pd.DataFrame(one_hot_encoder.fit_transform(target).toarray(), columns=diabetes_columns)
    return dataset, target


# Returns a train/validation split of the preprocessed dataset
# where all columns with ordinal values that are not simply yes/no are one hot encoded.
# If the parameter target_one_hot_encoded is set to true, the target column will also be one hot encoded.
# If the parameter include_test_data is set to true, test data will also be returned.
# This function does not include sampling.
def get_preprocessed_brfss_dataset_one_hot_encoded_train_test_split(target_one_hot_encoded=False,
                                                                    include_test_data=False) \
        -> Union[Tuple[DataFrame, DataFrame, DataFrame, DataFrame], Tuple[
            DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]]:
    dataset, target = get_preprocessed_brfss_dataset_one_hot_encoded(target_one_hot_encoded)
    return get_train_validation_test_split(dataset, target, include_test_data)


# Returns preprocessed dataset where all columns with ordinal values are one hot encoded (including yes/no columns).
# If the parameter is set to true, the target column will also be one hot encoded.
# This function does not include any sampling.
def get_preprocessed_brfss_dataset_one_hot_encoded_all_columns(target_one_hot_encoded=False) \
        -> Tuple[DataFrame, DataFrame]:
    dataset, target = get_preprocessed_brfss_dataset()
    one_hot_encoder = OneHotEncoder()
    encoded = pd.DataFrame(
        one_hot_encoder.fit_transform(dataset[all_ordinal_columns]).toarray(),
        columns=one_hot_encoder.get_feature_names_out())
    dataset = dataset.join(encoded)
    dataset = dataset.drop(columns=all_ordinal_columns)
    if target_one_hot_encoded:
        target = pd.DataFrame(one_hot_encoder.fit_transform(target).toarray(), columns=diabetes_columns)
    return dataset, target


# Returns a train/validation split of the preprocessed dataset
# where all columns with ordinal values are one hot encoded (including yes/no columns).
# If the parameter target_one_hot_encoded is set to true, the target column will also be one hot encoded.
# If the parameter include_test_data is set to true, test data will also be returned.
# This function does not include any sampling.
def get_preprocessed_brfss_dataset_one_hot_encoded_all_columns_train_test_split(target_one_hot_encoded=False,
                                                                                include_test_data=False) \
        -> Union[Tuple[DataFrame, DataFrame, DataFrame, DataFrame], Tuple[
            DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]]:
    dataset, target = get_preprocessed_brfss_dataset_one_hot_encoded_all_columns(target_one_hot_encoded)
    return get_train_validation_test_split(dataset, target, include_test_data)


# Returns a train/validation split of the preprocessed dataset
# where all columns with ordinal values that are not simply yes/no are one hot encoded.
# If the parameter target_one_hot_encoded is set to true, the target column will also be one hot encoded.
# If the parameter include_test_data is set to true, test data will also be returned.
# This function includes oversampling.
def get_preprocessed_brfss_dataset_one_hot_encoded_train_test_split_oversampled(target_one_hot_encoded=False,
                                                                                include_test_data=False) \
        -> Union[Tuple[DataFrame, DataFrame, DataFrame, DataFrame], Tuple[
            DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]]:
    if include_test_data:
        data_train, data_validation, data_test, target_train, target_validation, target_test = \
            get_preprocessed_brfss_dataset_one_hot_encoded_train_test_split(False, include_test_data)
        data_train, target_train = oversample_dataset(data_train, target_train)
        if target_one_hot_encoded:
            target_train = one_hot_encode_target(target_train)
            target_validation = one_hot_encode_target(target_validation)
            target_test = one_hot_encode_target(target_test)
        return data_train, data_validation, data_test, target_train, target_validation, target_test
    else:
        data_train, data_validation, target_train, target_validation = \
            get_preprocessed_brfss_dataset_one_hot_encoded_train_test_split(False, include_test_data)
        data_train, target_train = oversample_dataset(data_train, target_train)
        if target_one_hot_encoded:
            target_train = one_hot_encode_target(target_train)
            target_validation = one_hot_encode_target(target_validation)
        return data_train, data_validation, target_train, target_validation


# Returns a train/validation split of the preprocessed dataset
# where all columns with ordinal values that are not simply yes/no are one hot encoded.
# If the parameter target_one_hot_encoded is set to true, the target column will also be one hot encoded.
# If the parameter include_test_data is set to true, test data will also be returned.
# This function includes undersampling.
def get_preprocessed_brfss_dataset_one_hot_encoded_train_test_split_undersampled(target_one_hot_encoded=False,
                                                                                 include_test_data=False) \
        -> Union[Tuple[DataFrame, DataFrame, DataFrame, DataFrame], Tuple[
            DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]]:
    if include_test_data:
        data_train, data_validation, data_test, target_train, target_validation, target_test = \
            get_preprocessed_brfss_dataset_one_hot_encoded_train_test_split(False, include_test_data)
        data_train, target_train = undersample_dataset(data_train, target_train)
        if target_one_hot_encoded:
            target_train = one_hot_encode_target(target_train)
            target_validation = one_hot_encode_target(target_validation)
            target_test = one_hot_encode_target(target_test)
        return data_train, data_validation, data_test, target_train, target_validation, target_test
    else:
        data_train, data_validation, target_train, target_validation = \
            get_preprocessed_brfss_dataset_one_hot_encoded_train_test_split(False, include_test_data)
        data_train, target_train = undersample_dataset(data_train, target_train)
        if target_one_hot_encoded:
            target_train = one_hot_encode_target(target_train)
            target_validation = one_hot_encode_target(target_validation)
        return data_train, data_validation, target_train, target_validation


# Returns a train/validation split of the preprocessed dataset
# where all columns with ordinal values are one hot encoded (including yes/no columns).
# If the parameter target_one_hot_encoded is set to true, the target column will also be one hot encoded.
# If the parameter include_test_data is set to true, test data will also be returned.
# This function includes oversampling
def get_preprocessed_brfss_dataset_one_hot_encoded_all_columns_train_test_split_oversampled(
        target_one_hot_encoded=False, include_test_data=False) -> Union[
    Tuple[DataFrame, DataFrame, DataFrame, DataFrame], Tuple[
        DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]]:
    if include_test_data:
        data_train, data_validation, data_test, target_train, target_validation, target_test = \
            get_preprocessed_brfss_dataset_one_hot_encoded_all_columns_train_test_split(False, include_test_data)
        data_train, target_train = oversample_dataset(data_train, target_train)
        if target_one_hot_encoded:
            target_train = one_hot_encode_target(target_train)
            target_validation = one_hot_encode_target(target_validation)
            target_test = one_hot_encode_target(target_test)
        return data_train, data_validation, data_test, target_train, target_validation, target_test
    else:
        data_train, data_validation, target_train, target_validation = \
            get_preprocessed_brfss_dataset_one_hot_encoded_all_columns_train_test_split(False, include_test_data)
        data_train, target_train = oversample_dataset(data_train, target_train)
        if target_one_hot_encoded:
            target_train = one_hot_encode_target(target_train)
            target_validation = one_hot_encode_target(target_validation)
        return data_train, data_validation, target_train, target_validation


# Returns a train/validation split of the preprocessed dataset
# where all columns with ordinal values are one hot encoded (including yes/no columns).
# If the parameter target_one_hot_encoded is set to true, the target column will also be one hot encoded.
# If the parameter include_test_data is set to true, test data will also be returned.
# This function includes undersampling
def get_preprocessed_brfss_dataset_one_hot_encoded_all_columns_train_test_split_undersampled(
        target_one_hot_encoded=False, include_test_data=False) -> Union[
    Tuple[DataFrame, DataFrame, DataFrame, DataFrame], Tuple[
        DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]]:
    if include_test_data:
        data_train, data_validation, data_test, target_train, target_validation, target_test = \
            get_preprocessed_brfss_dataset_one_hot_encoded_all_columns_train_test_split(False, include_test_data)
        data_train, target_train = undersample_dataset(data_train, target_train)
        if target_one_hot_encoded:
            target_train = one_hot_encode_target(target_train)
            target_validation = one_hot_encode_target(target_validation)
            target_test = one_hot_encode_target(target_test)
        return data_train, data_validation, data_test, target_train, target_validation, target_test
    else:
        data_train, data_validation, target_train, target_validation = \
            get_preprocessed_brfss_dataset_one_hot_encoded_all_columns_train_test_split(False, include_test_data)
        data_train, target_train = undersample_dataset(data_train, target_train)
        if target_one_hot_encoded:
            target_train = one_hot_encode_target(target_train)
            target_validation = one_hot_encode_target(target_validation)
        return data_train, data_validation, target_train, target_validation


def one_hot_encode_target(target) -> DataFrame:
    one_hot_encoder = OneHotEncoder()
    return pd.DataFrame(one_hot_encoder.fit_transform(target).toarray(), columns=diabetes_columns)
