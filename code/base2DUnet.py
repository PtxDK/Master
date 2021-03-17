#!/usr/bin/env python3
#%%
print("loading imports")
from mpunet.models import UNet
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import *
from heartnet.data.base_loader import load2D
from heartnet.metrics.base import dice
#%%
print("setting up model")
N_CLASSES = 2
BATCH_SIZE = 16
EPOCHS = 500
net = UNet(N_CLASSES, depth=4, dim=128, out_activation="linear")
loss = SparseCategoricalCrossentropy(from_logits=True)
opt = Adam(5.0e-05, 0.9, 0.999, 1e-8, decay=0.0)
net.compile(opt, loss=loss, metrics=["accuracy", dice], run_eagerly=True)
train_ds = load2D("/homes/pmcd/Peter_Patrick3/train").batch(BATCH_SIZE)
val_ds = load2D("/homes/pmcd/Peter_Patrick3/val").batch(BATCH_SIZE)
test_ds = load2D("/homes/pmcd/Peter_Patrick3/test").batch(BATCH_SIZE)
cbs = [
    callbacks.ModelCheckpoint(
        "./model/base_2d_unet.h5",
        save_best_only=True,
        verbose=1,
        save_weights_only=True,
        monitor="val_dice",
        mode="max"
    ),
    callbacks.CSVLogger("./logs/base2DUNet.csv"),
    callbacks.ReduceLROnPlateau(
        patience=2, factor=0.90, verbose=1, monitor="val_dice", mode="max"
    ),
    callbacks.EarlyStopping(
        monitor='val_dice', min_delta=0, patience=15, verbose=1, mode='max'
    ),
]

#%%
print("starting training")
net.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=cbs,
    batch_size=BATCH_SIZE,
)
net.evaluate(test_ds)
# %%
