import matplotlib.pyplot as plt


def plot_loss(loss_values: [], title: str = "Loss curve"):
    plt.title(title)
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.plot(loss_values)
    plt.show()


def plot_multiple_loss_curves(loss_curves: [], labels: [], title: str = "Loss curve"):
    plt.title(title)
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    for index, loss in enumerate(loss_curves):
        plt.plot(loss, label=labels[index])
    plt.show()
