import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt


matplotlib.rcParams['figure.figsize'] = [9, 6]

def f(x):
    y = x**2 + 2*x - 5
    return y


x = tf.linspace(-2, 2, 201)
x = tf.cast(x, tf.float32)

noise = tf.random.normal(shape=tf.shape(x))

y = f(x) + noise

class Model(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=units,
                                            activation=tf.nn.relu,
                                            kernel_initializer=tf.random.normal,
                                            bias_initializer=tf.random.normal)
        self.dense2 = tf.keras.layers.Dense(1)
