import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from preprocessing.preprocessing import get_preprocessed_brfss_dataset, get_train_test_split, diabetes_columns, \
    undersample_dataset

CATEGORICAL_COLUMNS = ["GenHealth", "Healthcare", "MedCost", "Checkup", "HighBP",
                       "HighChol", "HeartAttack", "AngiCoro", "Stroke", "Asthma", "Arthritis", "Kidney", "Sex",
                       "Income", "SodiumSalt", "Age", "BMI", "Education", "Alcohol", "Smoking",
                       "FruitCons", "VegetCons", "PhysActivity", "Muscles"]
NUMERICAL_COLUMNS = ["PhysHealth", "MentHealth", "Height", "Weight"]


def get_number_of_numerical_features():
    return len(NUMERICAL_COLUMNS)


def get_number_of_categorical_features():
    return len(CATEGORICAL_COLUMNS)


class NeuralNetworkPreprocessor:

    def __init__(self):
        self.dataset, self.target = get_preprocessed_brfss_dataset()
        for column in CATEGORICAL_COLUMNS:
            self.dataset[column] = LabelEncoder().fit_transform(self.dataset[column])
            self.dataset[column] = self.dataset[column].astype("category")
            pass

    def get_embedding_sizes(self):
        categorical_column_sizes = [len(self.dataset[column].cat.categories) for column in CATEGORICAL_COLUMNS]
        categorical_embedding_sizes = [(column_size, min(50, (column_size + 1) // 2)) for column_size in
                                       categorical_column_sizes]
        return categorical_embedding_sizes

    def get_preprocessed_dataset_for_neural_network(self):
        one_hot_encoder = OneHotEncoder()
        self.target = pd.DataFrame(one_hot_encoder.fit_transform(self.target).toarray(), columns=diabetes_columns)
        data_train, data_test, target_train, target_test = get_train_test_split(self.dataset, self.target)
        return data_train, data_test, target_train, target_test

    def get_preprocessed_dataset_for_neural_network_undersampled(self):
        one_hot_encoder = OneHotEncoder()
        data_train, data_test, target_train, target_test = get_train_test_split(self.dataset, self.target)
        data_train, target_train = undersample_dataset(data_train, target_train)
        target_train = pd.DataFrame(one_hot_encoder.fit_transform(target_train).toarray(), columns=diabetes_columns)
        target_test = pd.DataFrame(one_hot_encoder.transform(target_test).toarray(), columns=diabetes_columns)
        return data_train, data_test, target_train, target_test
