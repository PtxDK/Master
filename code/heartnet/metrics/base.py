import tensorflow as tf
from mpunet.evaluate.metrics import dice_binary
def dice(true, pred):
    return dice_binary(true, tf.math.argmax(pred, axis=-1))