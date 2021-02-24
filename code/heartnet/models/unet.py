import tensorflow as tf
from tensorflow.keras.layers import *
from heartnet.layers import CropConcatLayer
from heartnet.layers.blocks import *


class UNet2D(tf.keras.Model):
    def __init__(self, num_classes, num_filters, depth, img_size: int,
                 convBlock=ConvBlock, upConvBlock=UpConvBlock, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.depth = depth
        self.img_size = img_size
        self.conv_block = convBlock
        self.up_conv_block = upConvBlock

    def create_model(self):
        inputs = Input(shape=(self.img_size, self.img_size, 1))
        x = inputs
        xs = []
        filters = self.num_filters
        for i in range(self.depth):
            x = self.conv_block(filters=filters, kernel_size=3, activation="relu")(x)
            filters *= 2
            if i < 4:
                xs.append(x)
                x = MaxPooling2D(strides=(2, 2))(x)
        for i in range(1, self.depth):
            filters /= 2
            x = self.up_conv_block(filters=filters, kernel_size=(2, 2), strides=(2, 2))(x)
            concat = CropConcatLayer(xs[-i], x)
            x = concat(xs[(-i)], x)
            x = self.conv_block(filters=filters, kernel_size=3, activation="relu")(x)
        output = Conv2D(self.num_classes, 1, activation="softmax")(x)
        return [inputs], [output]

    def init_model(self):
        pass

    def call(self):
        pass
