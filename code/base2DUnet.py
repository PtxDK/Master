#!/usr/bin/env python3
#%%
print("loading imports")
# from mpunet.evaluate.metrics import dice, sparse_mean_fg_f1
from mpunet.logging.default_logger import ScreenLogger
from mpunet.models import UNet
from mpunet.evaluate.metrics import *
from mpunet.callbacks import ValDiceScores
from mpunet.utils.utils import highlighted
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import *
import pathlib
import nibabel as nib
from tensorflow.python.keras.callbacks import Callback

#%%
print("setting up dataset")


class DiceScores(Callback):
    """
    Similar to Validation, but working on an array of data instead of
    internally sampling from a validation sequence generator.

    On epoch end computes the mean dice coefficient and adds to following log
    entry:
    logs["val_dice"] = mean_dice
    """

    def __init__(self, validation_data, n_classes, logger=None):
        """
        Args:
            validation_data: A tuple (X, y) of two ndarrays of validation data
                             and corresponding labels.
                             Any shape accepted by the model.
                             Labels must be integer targets (not one-hot)
            n_classes:       Number of classes, including background
            batch_size:      Batch size used for prediction
            logger:          An instance of a MultiPlanar Logger that prints to screen
                             and/or file
        """
        super().__init__()
        self.logger = logger or ScreenLogger()
        self.data = validation_data
        self.n_classes = n_classes
        self.scores = []

    def eval(self):
        dice = []
        for x, y in self.data:
            pred = self.model.predict(x, verbose=0)
            dices = dice_all(
                y, pred.argmax(-1), n_classes=self.n_classes, ignore_zero=True
            )
            dice.append(dices)
        return np.stack(dice)

    def on_epoch_end(self, epoch, logs={}):
        scores = self.eval()
        mean_dice = scores.mean()
        s = "Mean dice for epoch %d: %.4f\nPr. class: %s" % (
            epoch, mean_dice, scores[0, 0]
        )
        self.logger(highlighted(s))
        self.scores.append(mean_dice)

        # Add to log
        logs["val_dice"] = mean_dice


def load_img(flip):

    def func(x, y):
        ret_x = nib.load(x.numpy().decode("utf-8")).get_fdata()
        ret_y = nib.load(y.numpy().decode("utf-8")).get_fdata()
        ret_x = np.array([ret_x[:, :, i] for i in range(111)])
        ret_y = np.array([ret_y[:, :, i] for i in range(111)])
        return ret_x, ret_y

    return func


def base_loader(base_dir, split_slices=True):
    base_dir = pathlib.Path(base_dir)
    images = (base_dir / "images")
    labels = (base_dir / "labels")
    images_str = [str(i) for i in images.glob("*")]
    labels_str = [str(i) for i in labels.glob("*")]
    img_ds = tf.data.Dataset.from_tensor_slices(
        [str(i) for i in images.glob("*")]
    )
    label_ds = tf.data.Dataset.from_tensor_slices(
        [str(i) for i in labels.glob("*")]
    )
    dataset = tf.data.Dataset.zip((img_ds, label_ds))
    dataset = dataset.map(
        lambda x, y: tf.
        py_function(load_img(None), [x, y], [tf.float32, tf.int32]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.flat_map(
        lambda x, y: tf.data.Dataset.
        from_tensor_slices((x[:, :, :, None], y[:, :, :, None]))
    )
    return dataset


#%%
print("setting up model")
N_CLASSES = 2
net = UNet(N_CLASSES, depth=4, dim=128)
loss = SparseCategoricalCrossentropy(from_logits=True)
opt = Adam(5.0e-05, 0.9, 0.999, 1e-8, decay=0.0)
net.compile(opt, loss=loss, metrics=["accuracy"], run_eagerly=True)
dataset = base_loader("/homes/pmcd/Peter_Patrick3/train")
BATCH_SIZE = 16
EPOCHS = 500
DATASET_LEN = 46
IMAGES_PER_ENTRY = 111
train_dataset = dataset.take(int(0.8 * (DATASET_LEN*111))).batch(BATCH_SIZE)
val_dataset = dataset.skip(int(0.8 * (DATASET_LEN*111))).batch(BATCH_SIZE)
cbs = [
    DiceScores(val_dataset, N_CLASSES),
    callbacks.ReduceLROnPlateau(
        patience=2, factor=0.90, verbose=1, monitor="val_dice", mode="max"
    ),
    callbacks.ModelCheckpoint(
        "./model/base_2d_unet.h5",
        save_best_only=True,
        save_weights_only=True,
        monitor="val_dice",
        mode="max"
    ),
    callbacks.EarlyStopping(
        monitor='val_dice', min_delta=0, patience=15, verbose=1, mode='max'
    )
]

#%%
print("starting training")
net.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=cbs,
    batch_size=BATCH_SIZE,
)

# %%
