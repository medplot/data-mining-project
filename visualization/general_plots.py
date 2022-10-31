from pandas import DataFrame
import matplotlib.pyplot as plt


def plot_class_frequencies(feature: DataFrame):
    feature = feature.squeeze()
    class_dist = feature.value_counts()
    plt.bar(class_dist.index, class_dist)
    plt.ylabel("Frequency")
    plt.xlabel("Class")
    plt.show()
