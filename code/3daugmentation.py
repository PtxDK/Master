#!/usr/bin/env python3
from mpunet.models import UNet3D
from heartnet.models.base import BaseModelTraining
from heartnet.augmentation.elastic import Elastic3D
base3D = BaseModelTraining(
    UNet3D(2, dim=112, out_activation="softmax"), name="augmentation"
)
base3D.batch_size = 1
base3D.augmentations = [
    Elastic3D(alpha=[0, 450], sigma=[20, 30], apply_prob=0.333)
]
base3D.setup(True)    
# base3D.train()
base3D.evaluate()