#%%
import tensorflow as tf
from tensorflow.python.keras.backend import dropout
from heartnet.layers.patches import *

from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow_addons as tfa

# mlp_head_units = [2048, 1024]


#%%
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units[:-1]:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(hidden_units[-1])(x)
    x = layers.Dropout(dropout_rate)(x)
    return x


def create_transformer_module(num_patches, num_transformers, num_heads,
                              projection_dim, transformer_units, dropout_rate,
                              attn_dropout):

    # input_shape: [1, latent_dim, projection_dim]
    inputs = layers.Input(shape=(num_patches, projection_dim), batch_size=8)

    x0 = inputs
    # Create multiple layers of the Transformer block.
    for _ in range(num_transformers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(x0)
        # print(x1.shape, inputs.shape)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(num_heads=num_heads,
                                                     key_dim=projection_dim,
                                                     dropout=attn_dropout)(x1,
                                                                           x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, x0])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=dropout_rate)
        # Skip connection 2.
        x0 = layers.Add()([x3, x2])

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=x0)
    return model


class TransUnet(Model):
    def __init__(
        self,
        n_classes,
        input_shape,
        patch_size,
        projection_dim,
        num_heads,
        num_transformer_blocks,
        mlp_units,
        dropout_rate,
        attn_dropout_rate,
        num_iterations,
    ):
        super(TransUnet, self).__init__()
        self.n_classes = n_classes
        # self.input_shape = input_shape

        self.num_patches = (input_shape // patch_size)**2
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.num_transformer_blocks = num_transformer_blocks
        self.mlp_units = mlp_units
        self.dropout_rate = dropout_rate
        self.attn_dropout = attn_dropout_rate
        self.num_iterations = num_iterations

    def build(self, input_shape):

        # Create patching module.
        self.patcher = Patches(self.patch_size)

        # Create patch encoder.
        self.patch_encoder = PatchEncoder(self.num_patches,
                                          self.projection_dim)

        # Create Transformer module.
        self.transformer = create_transformer_module(
            self.num_patches, self.num_transformer_blocks, self.num_heads,
            self.projection_dim, [
                self.mlp_units,
                self.projection_dim,
            ], self.dropout_rate, self.attn_dropout)

        super().build(input_shape)

    def call(self, x):
        patches = self.patcher(x)
        encoded_patches = self.patch_encoder(patches)
        return self.transformer(encoded_patches)


transformer = TransUnet(2, 128, 16, 768, 12, 12, 3072, 0.1, 0.0, 10)
#%%
transformer(tf.ones([1, 128, 128, 1]))
# %%
