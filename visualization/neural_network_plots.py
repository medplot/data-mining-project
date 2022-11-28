import matplotlib.pyplot as plt
from datetime import datetime


def plot_loss(loss_values: [], title: str = "Loss curve"):
    fig = plt.figure()
    plt.title(title)
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.plot(loss_values)
    plt.show()
    fig.savefig(f"Neural network loss {datetime.now()}.png", dpi=300)


def plot_multiple_loss_curves(loss_curves: [], labels: [], title: str = "Loss curve"):
    fig = plt.figure()
    plt.title(title)
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    for index, loss in enumerate(loss_curves):
        plt.plot(loss, label=labels[index])
    plt.legend()
    fig.savefig(f"Neural network loss curves {datetime.now()}.png", dpi=300)


def plot_accuracy(accuracy_values: [], title: str = "Accuracy curve"):
    fig = plt.figure()
    plt.title(title)
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.plot(accuracy_values)
    plt.show()
    fig.savefig(f"Neural network accuracy {datetime.now()}.png", dpi=300)


def plot_multiple_accuracy_curves(accuracy_curves: [], labels: [], title: str = "Accuracy curve"):
    fig = plt.figure()
    plt.title(title)
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    for index, loss in enumerate(accuracy_curves):
        plt.plot(loss, label=labels[index])
    plt.legend()
    fig.savefig(f"Neural network accuracies {datetime.now()}.png", dpi=300)


def plot_multiple_accuracy_curves(accuracy_curves: [], labels: [], title: str = "Accuracy curve"):
    fig = plt.figure()
    plt.title(title)
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    for index, loss in enumerate(accuracy_curves):
        plt.plot(loss, label=labels[index])
    plt.legend()
    fig.savefig(f"Neural network accuracies {datetime.now()}.png", dpi=300)


def plot_multiple_f_scores(f_scores: [], labels: [], title: str = "F-score curve"):
    fig = plt.figure()
    plt.title(title)
    plt.ylabel("F-score")
    plt.xlabel("Epochs")
    for index, loss in enumerate(f_scores):
        plt.plot(loss, label=labels[index])
    plt.legend()
    fig.savefig(f"Neural network f-scores {datetime.now()}.png", dpi=300)
