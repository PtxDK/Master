#!/usr/bin/env python3
#%%
print("loading imports")
from mpunet.models import UNet, UNet3D
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import *
from heartnet.data.base_loader import load3D, base_loader, crop_image
from heartnet.metrics.base import dice
import tensorflow as tf
N_CLASSES = 2
BATCH_SIZE = 1
EPOCHS = 500
IMG_SIZE = 96
net = UNet3D(N_CLASSES, dim=IMG_SIZE, out_activation="linear")


#%%


#%%
print("setting up model")

loss = SparseCategoricalCrossentropy(from_logits=True)
opt = Adam(5.0e-05, 0.9, 0.999, 1e-8, decay=0.0)
net.compile(opt, loss=loss, metrics=["accuracy", dice], run_eagerly=True)
train_ds = load3D("/homes/pmcd/Peter_Patrick3/train",
                  IMG_SIZE).batch(BATCH_SIZE)
val_ds = load3D("/homes/pmcd/Peter_Patrick3/val", IMG_SIZE).batch(BATCH_SIZE)
test_ds = load3D("/homes/pmcd/Peter_Patrick3/test", IMG_SIZE).batch(BATCH_SIZE)
cbs = [
    callbacks.ModelCheckpoint(
        "./model/base_3d_unet.h5",
        save_best_only=True,
        verbose=1,
        save_weights_only=True,
        monitor="val_dice",
        mode="max"
    ),
    callbacks.CSVLogger("./logs/base3DUNet.csv"),
    callbacks.ReduceLROnPlateau(
        patience=2, factor=0.90, verbose=1, monitor="val_dice", mode="max"
    ),
    callbacks.EarlyStopping(
        monitor='val_dice', min_delta=0, patience=15, verbose=1, mode='max'
    ),
]

#%%
for x,y in train_ds.take(1):
    print(x.shape, y.shape)
# print("starting training")
net.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=cbs,
    batch_size=BATCH_SIZE,
)
net.evaluate(test_ds)

# %%
