import tensorflow as tf
import tensorflow.keras.datasets as tfds
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.activations import relu


def load_data():
    (x_train, y_train), (x_test, y_test) = tfds.mnist.load_data()
    # normalize x
    x_train = x_train.astype(float) / 255.
    x_test = x_test.astype(float) / 255.
    return (x_train, y_train), (x_test, y_test)


class MnistModel(Model):

    def get_config(self):
        super(MnistModel, self).get_config()

    def __init__(self, inputs):
        super(MnistModel, self).__init__()

        self.flatten = Flatten()
        self.dense1 = Dense(784, activation=relu)
        self.dense2 = Dense(128, activation=relu)
        self.classifier = tf.keras.layers.Dense(10, activation='softmax', name='predictions')

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.classifier(x)

    def my_compile(self, metrics='accuracy'):
        self.compile(optimizer=tf.keras.optimizers.RMSprop(),
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=metrics)

    def my_fit(self, x, y, batch_size=10, epochs=1, verbose=1, validation_split=0.1):
        return self.fit(x=x, y=y,
                        batch_size=batch_size, epochs=epochs,
                        verbose=verbose, validation_split=validation_split)
