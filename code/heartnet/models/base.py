from heartnet.loader.base_loader import *
from ..metrics.classes import *
from tensorflow.keras import *


class BaseModelTraining(object):

    def __init__(self, model, loss) -> None:
        super().__init__()
        self.name = "base"
        self.num_slices = 111
        self.image_size = (128, 128)
        self.data_base_folder = "/homes/pmcd/Peter_Patrick3"
        self.data_train_folder = "train"
        self.data_val_folder = "val"
        self.data_test_folder = "test"
        self.batch_size = 16
        self.metrics = [Dice(), FGF1Score(), FGRecall(), FGPrecision()]
        self.model = model

        self.loss = loss or SparseCategoricalCrossentropy()
        self.epochs = 500
        mult = self.batch_size / 16 if self.batch_size > 16 else 1
        self.optimizer = optimizers.Adam(
            1e-4 * (mult), 0.9, 0.999, 1e-8, decay=0.0
        )
        self._file_name = f"{self.model_name}_{self.name}"
        self.callbacks = [
            callbacks.CSVLogger(f"./logs/{self._file_name}.csv"),
            callbacks.ReduceLROnPlateau(
                patience=2,
                factor=0.90,
                verbose=1,
                monitor="val_dice",
                mode="max"
            ),
            callbacks.EarlyStopping(
                monitor='val_dice',
                min_delta=0,
                patience=15,
                verbose=1,
                mode='max'
            ),
            callbacks.ModelCheckpoint(
                f"./model/{self._file_name}.h5",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
                monitor="val_dice",
                mode="max"
            ),
        ]
        self._train_ds, self._val_ds, self._test_ds = None, None, None

    @property
    def model_name(self):
        return self.model.__class__.__name__

    def setup(self):
        self._train_ds, self._val_ds, self._test_ds = self.load_datasets()
        self.model.compile(self.optimizer, self.loss, metrics=self.metrics)

    def train(self):
        pass

    def evaluate(self):
        pass

    def load_datasets(self):
        load_function = load_functions[self.model_name]
        splits = [
            self.data_train_folder, self.data_val_folder, self.data_test_folder
        ]
        ret = {i: None for i in splits}
        base_folder = pathlib.Path(self.data_base_folder)
        for split in splits:
            ds = load_function(base_folder / split)
            ds = ds.batch(self.batch_size)
            # ds = ds.map(
            #     lambda x, y: tf.py_function(
            #         apply_augmentations(config["data"]["augmentations"]), [x, y],
            #         [tf.float32, tf.float32]
            #     )
            # )
            ret[split] = ds
        return list(ret.values())