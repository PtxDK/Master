import tensorflow as tf
from tensorflow.keras.metrics import *
from tensorflow.python.keras.metrics import MeanMetricWrapper
from mpunet.evaluate.metrics import one_class_dice, sparse_fg_recall, sparse_fg_precision


class Dice(MeanMetricWrapper):

    def __init__(self, name="dice", dtype=None, **kwargs):
        # smooth = tf.constant(smooth, tf.int64)

        def fn(true, pred):
            pred = tf.cast(tf.argmax(pred, axis=-1), tf.float32)
            return one_class_dice(tf.squeeze(true), pred)

        super().__init__(fn, name=name, dtype=dtype, **kwargs)


class FGRecall(MeanMetricWrapper):

    def __init__(self, name="fg_recall", dtype=None, **kwargs):

        def fn(true, pred):
            return sparse_fg_recall(true, pred, 0)

        super().__init__(fn, name=name, dtype=dtype, **kwargs)


class FGPrecision(MeanMetricWrapper):

    def __init__(self, name="fg_precision", dtype=None, **kwargs):

        def fn(true, pred):
            return sparse_fg_precision(true, pred, 0)

        super().__init__(fn, name=name, dtype=dtype, **kwargs)


class FGF1Score(MeanMetricWrapper):

    def __init__(self, name="fg_f1", dtype=None, **kwargs):

        def fn(y_true, y_pred):
            y_pred = tf.argmax(y_pred, axis=-1)

            # Get confusion matrix
            cm = tf.math.confusion_matrix(
                tf.reshape(y_true, [-1]), tf.reshape(y_pred, [-1])
            )

            # Get precisions
            TP = tf.linalg.diag_part(cm)
            precisions = TP / tf.reduce_sum(cm, axis=0)

            # Get recalls
            TP = tf.linalg.diag_part(cm)
            recalls = TP / tf.reduce_sum(cm, axis=1)

            # Get F1s
            f1s = (2*precisions*recalls) / (precisions+recalls)

            return tf.math.reduce_mean(f1s[1:])

        super().__init__(fn, name=name, dtype=dtype, **kwargs)