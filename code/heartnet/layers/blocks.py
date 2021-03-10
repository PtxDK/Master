from tensorflow.keras import layers
from tensorflow.keras.layers import *


class ConvBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.conv1 = Conv2D(filters, kernel_size, activation=activation)
        self.conv2 = Conv2D(filters, kernel_size, activation=activation)

    def call(self, x):
        x = self.conv1(x)
        print(x.shape)
        return self.conv2(x)


class UpConvBlock(layers.Layer):
    def __init__(self, filters, kernel_size=(2,2), strides=(2,2), **kwargs):
        super().__init__(**kwargs)
        self.up_conv = Conv2DTranspose(filters, kernel_size,
                            strides=strides)

    def call(self, x):
        return self.up_conv(x)