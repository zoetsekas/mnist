# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input
import numpy as np
from model import MnistModel, load_data

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    inputs = Input(shape=(28, 28))
    mnist = MnistModel(inputs=inputs)

    # plot_model(mnist, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    (x_train, y_train), (x_test, y_test) = load_data()

    x_pred = x_train[:15]
    y_pred = y_train[:15]

    mnist.my_compile()
    mnist.my_fit(x=x_train, y=y_train)
    mnist.summary()
    acc = mnist.evaluate(x=x_test, y=y_test)
    print(acc)

    y_hat = mnist.predict(x_pred)

    print(np.argmax(y_hat))

    # # Plot a random sample of 10 test images, their predicted labels and ground truth
    # figure = plt.figure(figsize=(20, 8))
    # for i, index in enumerate(np.random.choice(x_pred.shape[0], size=10, replace=False)):
    #     ax = figure.add_subplot(2, 5, i + 1, xticks=[], yticks=[])
    #     # Display each image
    #     ax.imshow(np.squeeze(x_pred[index]))
    #     predict_index = np.argmax(y_hat[index])
    #     true_index = np.argmax(y_pred[index])
    #     # Set the title for each image
    #     ax.set_title("{} ({})".format(predict_index,
    #                                   true_index),
    #                  color=("green" if predict_index == true_index else "red"))
