from preprocessing.preprocessing_label_encoding import get_preprocessed_brfss_dataset_label_encoded_train_test_split, \
    get_preprocessed_brfss_dataset_label_encoded_train_test_split_oversampled, \
    get_preprocessed_brfss_dataset_label_encoded_train_test_split_undersampled
from preprocessing.preprocessing_one_hot_encoding import \
    get_preprocessed_brfss_dataset_one_hot_encoded_train_test_split, \
    get_preprocessed_brfss_dataset_one_hot_encoded_train_test_split_oversampled, \
    get_preprocessed_brfss_dataset_one_hot_encoded_train_test_split_undersampled, \
    get_preprocessed_brfss_dataset_one_hot_encoded_all_columns_train_test_split, \
    get_preprocessed_brfss_dataset_one_hot_encoded_all_columns_train_test_split_oversampled, \
    get_preprocessed_brfss_dataset_one_hot_encoded_all_columns_train_test_split_undersampled


class TestPreprocessing:

    def test_get_preprocessed_brfss_dataset_label_encoded_train_test_split(self):
        train_data, test_data, train_target, test_target = \
            get_preprocessed_brfss_dataset_label_encoded_train_test_split()
        assert train_data is not None, "train_data not None"
        assert test_data is not None, "test_data not None"
        assert train_target is not None, "train_target not None"
        assert test_target is not None, "test_target not None"

    def test_get_preprocessed_brfss_dataset_label_encoded_train_test_split_oversampled(self):
        train_data, test_data, train_target, test_target = \
            get_preprocessed_brfss_dataset_label_encoded_train_test_split_oversampled()
        assert train_data is not None, "train_data not None"
        assert test_data is not None, "test_data not None"
        assert train_target is not None, "train_target not None"
        assert test_target is not None, "test_target not None"

    def test_get_preprocessed_brfss_dataset_label_encoded_train_test_split_undersampled(self):
        train_data, test_data, train_target, test_target = \
            get_preprocessed_brfss_dataset_label_encoded_train_test_split_undersampled()
        assert train_data is not None, "train_data not None"
        assert test_data is not None, "test_data not None"
        assert train_target is not None, "train_target not None"
        assert test_target is not None, "test_target not None"

    def test_get_preprocessed_brfss_dataset_one_hot_encoded_train_test_split(self):
        train_data, test_data, train_target, test_target = \
            get_preprocessed_brfss_dataset_one_hot_encoded_train_test_split()
        assert train_data is not None, "train_data not None"
        assert test_data is not None, "test_data not None"
        assert train_target is not None, "train_target not None"
        assert test_target is not None, "test_target not None"

    def test_get_preprocessed_brfss_dataset_one_hot_encoded_train_test_split_oversampled(self):
        train_data, test_data, train_target, test_target = \
            get_preprocessed_brfss_dataset_one_hot_encoded_train_test_split_oversampled()
        assert train_data is not None, "train_data not None"
        assert test_data is not None, "test_data not None"
        assert train_target is not None, "train_target not None"
        assert test_target is not None, "test_target not None"

    def test_get_preprocessed_brfss_dataset_one_hot_encoded_train_test_split_undersampled(self):
        train_data, test_data, train_target, test_target = \
            get_preprocessed_brfss_dataset_one_hot_encoded_train_test_split_undersampled()
        assert train_data is not None, "train_data not None"
        assert test_data is not None, "test_data not None"
        assert train_target is not None, "train_target not None"
        assert test_target is not None, "test_target not None"

    def test_get_preprocessed_brfss_dataset_one_hot_encoded_all_columns_train_test_split(self):
        train_data, test_data, train_target, test_target = \
            get_preprocessed_brfss_dataset_one_hot_encoded_all_columns_train_test_split()
        assert train_data is not None, "train_data not None"
        assert test_data is not None, "test_data not None"
        assert train_target is not None, "train_target not None"
        assert test_target is not None, "test_target not None"

    def test_get_preprocessed_brfss_dataset_one_hot_encoded_all_columns_train_test_split_oversampled(self):
        train_data, test_data, train_target, test_target = \
            get_preprocessed_brfss_dataset_one_hot_encoded_all_columns_train_test_split_oversampled()
        assert train_data is not None, "train_data not None"
        assert test_data is not None, "test_data not None"
        assert train_target is not None, "train_target not None"
        assert test_target is not None, "test_target not None"

    def test_get_preprocessed_brfss_dataset_one_hot_encoded_all_columns_train_test_split_undersampled(self):
        train_data, test_data, train_target, test_target = \
            get_preprocessed_brfss_dataset_one_hot_encoded_all_columns_train_test_split_undersampled()
        assert train_data is not None, "train_data not None"
        assert test_data is not None, "test_data not None"
        assert train_target is not None, "train_target not None"
        assert test_target is not None, "test_target not None"
