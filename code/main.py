#!/usr/bin/env python3
#%%
print("loading imports")
# from mpunet.evaluate.metrics import dice, sparse_mean_fg_f1
from mpunet.models import UNet
from mpunet.evaluate.metrics import *
from mpunet.callbacks import ValDiceScores
import numpy as np
from heartnet.data.base_loader import base_loader
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import *
import pathlib
import nibabel as nib

#%%
print("setting up dataset")


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
net = UNet(2, depth=4, dim=128)
# ms = [metrics.CategoricalAccuracy(), dice, sparse_mean_fg_f1]
loss = SparseCategoricalCrossentropy(from_logits=True)
opt = Adam(5.0e-05, 0.9, 0.999, 1e-8, decay=0.0)
net.compile(opt, loss=loss, metrics=[], run_eagerly=True)
dataset = base_loader("/homes/pmcd/Peter_Patrick3/train")
BATCH_SIZE = 16
EPOCHS = 500
DATASET_LEN = 46
IMAGES_PER_ENTRY = 111
cbs = [
    callbacks.ReduceLROnPlateau(
        patience=2, factor=0.90, verbose=1, monitor="val_loss", mode="max"
    ),
    callbacks.
    ModelCheckpoint("./model/base_2d_unet_{epoch:02d}_loss_{loss:02f}.h5"),
    callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='max'
    ),
]
train_dataset = dataset.take(int(0.8 * (DATASET_LEN*111)))
val_dataset = dataset.skip(int(0.8 * (DATASET_LEN*111)))
#%%
print("starting training")
net.fit(
    train_dataset.batch(BATCH_SIZE),
    validation_data=val_dataset.batch(BATCH_SIZE),
    epochs=EPOCHS,
    callbacks=cbs,
    batch_size=BATCH_SIZE,
)
