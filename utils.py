import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# utility to display a row of images with their predictions and true labels
def display_images(image, predictions, labels, title, n):
    display_strings = [str(i) + "\n\n" + str(j) for i, j in zip(predictions, labels)]

    plt.figure(figsize=(17, 3))
    plt.title(title)
    plt.yticks([])
    plt.xticks([28 * x + 14 for x in range(n)], display_strings)
    plt.grid(None)
    image = np.reshape(image, [n, 28, 28])
    image = np.swapaxes(image, 0, 1)
    image = np.reshape(image, [28, 28 * n])
    plt.imshow(image)


def plot_metrics(train_metric, val_metric, metric_name, title, ylim=5):
    plt.title(title)
    plt.ylim(0, ylim)
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.plot(train_metric, color='blue', label=metric_name)
    plt.plot(val_metric, color='green', label='val_' + metric_name)

# plot_metrics(epochs_train_losses, epochs_val_losses, "Loss", "Loss", ylim=1.0)
