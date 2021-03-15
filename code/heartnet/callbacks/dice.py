from mpunet.logging.default_logger import ScreenLogger
from mpunet.models import UNet
from mpunet.evaluate.metrics import *
from mpunet.utils.utils import highlighted
from tensorflow.keras import *
from tensorflow.python.keras.callbacks import Callback
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