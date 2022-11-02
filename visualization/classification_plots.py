import itertools
import matplotlib.pyplot as plt
import numpy as np
import math


def plot_decision_boundary(df, target, estimator):
    # create a list of all columns that we are considering
    features = df.columns

    # create all combinations of considered columns
    combinations = list(itertools.combinations(features, 2))

    # create a figure and specify its size
    # fig = plt.figure(figsize=(15, 20))

    # go through all combinations and create one plot for each
    figure_index = 1
    plot_step = 0.02
    cols = 3
    rows = math.ceil(len(combinations) / cols)
    for combination in combinations:
        # Plot the decision boundary
        plt.subplot(rows, cols, figure_index)
        figure_index += 1

        x_min, x_max = df[combination[0]].min() - 1, df[combination[0]].max() + 1
        y_min, y_max = df[combination[1]].min() - 1, df[combination[1]].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

        estimator.fit(df[[combination[0], combination[1]]], target)
        z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)
        plt.pcolormesh(xx, yy, z, cmap=plt.cm.RdYlBu, shading='auto')

        plt.xlabel(combination[0])
        plt.ylabel(combination[1])

        for cls in set(target):
            group = df[target == cls]
            # plot the data points for the current group and feature combination
            plt.scatter(group[combination[0]], group[combination[1]], label=cls, edgecolor='black')


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
