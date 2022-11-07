from preprocessing.preprocessing import get_train_test_split, get_preprocessed_brfss_dataset
from preprocessing.preproccessing_one_hot_encoding import get_preprocessed_brfss_dataset_one_hot_encoded


class TestPreprocessing:

    def test_get_preprocessed_train_test_split(self):
        dataset, target = get_preprocessed_brfss_dataset()
        train_data, test_data, train_target, test_target = get_train_test_split(dataset, target)
        assert train_data is not None, "train_data not None"
        assert test_data is not None, "test_data not None"
        assert train_target is not None, "train_target not None"
        assert test_target is not None, "test_target not None"

    def test_get_preprocessed_brfss_dataset_one_hot_encoded(self):
        dataset, target = get_preprocessed_brfss_dataset_one_hot_encoded()
        assert dataset is not None, "dataset not None"
        assert target is not None, "target not None"
