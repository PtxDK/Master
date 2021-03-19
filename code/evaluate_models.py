from heartnet.config import YamlConfig
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import *
from heartnet.loader import load_datasets
import sys
yaml_files = sys.argv[1:]
configs = [YamlConfig.from_file(file) for file in yaml_files]
for i in configs:
    print("running model")
    print(i.config)
    config = i.config
    mult = config["fit"]["batch_size"] / 16 if config["fit"]["batch_size"] > 16 else 1
    opt = Adam(
        1e-4 * (mult), 0.9, 0.999, 1e-8, decay=0.0
    )
    net = config["model"]["model"]
    ms = config["model"]["metrics"]
    net.compile(opt, config["model"]["loss"], metrics=ms)
    train_ds, val_ds, test_ds = load_datasets(config)
    file_name = f"{config['model']['model'].__class__.__name__}_{config['name']}"
    cbs = [
        callbacks.CSVLogger(f"./logs/{file_name}.csv"),
        callbacks.ReduceLROnPlateau(
            patience=2, factor=0.90, verbose=1, monitor="val_dice", mode="max"
        ),
        callbacks.EarlyStopping(
            monitor='val_dice', min_delta=0, patience=15, verbose=1, mode='max'
        ),
        callbacks.ModelCheckpoint(
            f"./model/{file_name}.h5",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
            monitor="val_dice",
            mode="max"
        ),
    ]

    net.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["fit"]["epochs"],
        callbacks=cbs,
    )
    if test_ds:
        net.evaluate(test_ds)