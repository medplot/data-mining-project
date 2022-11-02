from preprocessing.preprocessing import get_preprocessed_brfss_train_test_split


class TestPreprocessing:

    def test_get_preprocessed_train_test_split(self):
        train_data, test_data, train_target, test_target = get_preprocessed_brfss_train_test_split()
        assert train_data is not None, "train_data not None"
        assert test_data is not None, "test_data not None"
        assert train_target is not None, "train_target not None"
        assert test_target is not None, "test_target not None"
