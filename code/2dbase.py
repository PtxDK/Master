#!/usr/bin/env python3
from mpunet.models import UNet
from heartnet.models.base import BaseModelTraining
from tensorflow.keras import *

base = BaseModelTraining(UNet(2, depth=4, dim=128, out_activation="softmax"))
base.batch_size = 16
base.setup()
base.train()
base.evaluate()