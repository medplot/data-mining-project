import matplotlib.pyplot as plt
from datetime import datetime


def plot_loss(loss_values: [], title: str = "Loss curve"):
    fig = plt.figure()
    plt.title(title)
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.plot(loss_values)
    plt.show()
    fig.savefig(f"Neural network loss {datetime.now()}.png", dpi=fig.dpi)


def plot_multiple_loss_curves(loss_curves: [], labels: [], title: str = "Loss curve"):
    plt.title(title)
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    for index, loss in enumerate(loss_curves):
        plt.plot(loss, label=labels[index])
    plt.show()


def plot_accuracy(accuracy_values: [], title: str = "Accuracy curve"):
    fig = plt.figure()
    plt.title(title)
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.plot(accuracy_values)
    plt.show()
    fig.savefig(f"Neural network accuracy {datetime.now()}.png", dpi=fig.dpi)
