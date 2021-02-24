from tensorflow.keras import *


class CropConcatLayer(layers.Layer):
    def __init__(self, left, right, **kwargs):
        super().__init__(**kwargs)
        left_shape = left.shape
        right_shape = right.shape
        diff = left_shape[1] - right_shape[1]
        self.crop_amount = diff // 2
        self.concat = layers.Concatenate(axis=-1)

    def call(self, left, right):
        left_sliced = left[:, self.crop_amount:-self.crop_amount,
                           self.crop_amount:-self.crop_amount, :]
        return self.concat([left_sliced, right])