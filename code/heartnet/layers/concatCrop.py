from tensorflow.keras import *


class CropConcatLayer(layers.Layer):
    def __init__(self, left, right, **kwargs):
        super().__init__(**kwargs)
        left_shape = left.shape
        right_shape = right.shape
        diff = left_shape[1] - right_shape[1]
        print(diff)