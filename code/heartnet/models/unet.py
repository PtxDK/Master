import tensorflow as tf
from tensorflow.keras.layers import *
from ..layers import CropConcatLayer
from ..layers.blocks import *


class UNet2D(tf.keras.Model):

    def __init__(
        self,
        num_classes,
        num_filters=64,
        depth=5,
        img_size: int=572,
        convBlock=ConvBlock,
        upConvBlock=UpConvBlock,
        **kwargs
    ) -> None:
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.depth = depth
        self.img_size = img_size
        self.conv_block = convBlock
        self.up_conv_block = upConvBlock
        self._model = None
        super().__init__(*self.create_model())

    def create_model(self):
        inputs = Input(shape=(self.img_size, self.img_size, 1))
        x = inputs
        xs = []
        filters = self.num_filters
        for i in range(self.depth):
            print(x.shape)
            x = self.conv_block(
                filters=filters, kernel_size=3, activation="relu"
            )(x)
            filters *= 2
            if i < 4:
                xs.append(x)
                x = MaxPooling2D(strides=(2, 2))(x)
        for i in range(1, self.depth):
            print(x.shape)
            filters /= 2
            x = self.up_conv_block(
                filters=filters, kernel_size=(2, 2), strides=(2, 2)
            )(x)
            concat = CropConcatLayer(xs[-i], x)
            x = concat(xs[(-i)], x)
            x = self.conv_block(
                filters=filters, kernel_size=3, activation="relu"
            )(x)
        x = Conv2D(self.num_classes, 1, activation="relu")(x)
        outputs = Activation("softmax")(x)
        print(inputs, outputs)
        return [inputs], [outputs]

    def compile_model(self):
        opt = tf.keras.optimizers.Adam
        # self._model = tf.keras.Model(inputs=inputs, outputs=output)
        self._model.compile()
