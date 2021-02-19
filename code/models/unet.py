#%%
import tensorflow as tf
from tensorflow.keras.layers import *

# from tensorflow import keras as K


#%%
class UNet2D(tf.keras.Model):
    def __init__(self, num_classes, num_filters, depth, img_size: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.depth = depth
        self.img_size = img_size

    def create_model(self):
        inputs = Input(shape=(self.img_size, self.img_size, 1))
        x = inputs
        xs = []
        filters = self.num_filters
        for i in range(self.depth):
            print(x.shape)
            x = Conv2D(filters, 3, activation="relu")(x)
            print(x.shape)
            x = Conv2D(filters, 3, activation="relu")(x)
            print(x.shape)
            filters *= 2
            if i < 4:
                xs.append(x)
                x = MaxPooling2D(strides=(2, 2))(x)
        for i in range(1, self.depth):
            x = UpSampling2D()(x)
            print(x.shape)
            concat = Concatenate()
            x = concat([
                xs[(-i)], x
            ])
            filters /= 2
            print(x.shape)
            x = UpSampling2D()(x)
            print(x.shape)
        output = Conv2D(2, 1)(x)
        return [inputs], [output]

    def init_model(self):
        pass

    def call(self):
        pass


#%%
net = UNet2D(2, 64, 5, 572)
net.create_model()