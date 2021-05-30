from mpunet.models import UNet3D, UNet
from heartnet.models.base import BaseModelTraining
for i in range(3):
    base = BaseModelTraining(
        UNet(2, depth=4, dim=128, out_activation="softmax", complexity_factor=2),
        f"base{i}"
    )
    base.batch_size = 16
    base.setup(True)
    base.save_output("train")
    base.save_output("val")
    base.save_output("test")
    base.save_output("final")
    base3D = BaseModelTraining(
        UNet3D(2, dim=112, out_activation="softmax"), name=f"pad{i}"
    )
    base3D.batch_size = 1
    base3D.setup(True)
    base3D.save_output("train")
    base3D.save_output("val")
    base3D.save_output("test")
    base3D.save_output("final")