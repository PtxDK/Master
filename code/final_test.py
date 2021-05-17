from mpunet.models import UNet3D, UNet
from heartnet.models.base import BaseModelTraining
for i in range(3):
    # base3D = BaseModelTraining(
    #     UNet3D(2, dim=112, out_activation="softmax"), name=f"pad{i}", full=True
    # )
    # base3D.batch_size = 1
    # base3D.setup(True)
    # base3D.evaluate()
    # base3D = BaseModelTraining(
    #     UNet3D(2, dim=96, out_activation="softmax"), name=f"shrink{i}", full=True
    # )
    # base3D.batch_size = 1
    # base3D.setup(True)
    # base3D.evaluate()
    # base3D = BaseModelTraining(
    #     UNet3D(2, dim=96, out_activation="softmax"),
    #     name=f"aug-96-0.333-100-{i}",
    #     full=True
    # )
    # base3D.batch_size = 1
    # base3D.setup(True)
    # base3D.evaluate()
    base3D = BaseModelTraining(
        UNet3D(2, dim=112, out_activation="softmax"),
        name=f"augmentation-shake-0.3332",
        full=True
    )
    base3D.batch_size = 1
    base3D.setup(True)
    base3D.evaluate()
    # base = BaseModelTraining(
    #     UNet(2, depth=4, dim=128, out_activation="softmax", complexity_factor=2),
    #     f"base{i}",
    #     full=True
    # )
    # base.batch_size = 16
    # base.setup(True)
    # base.evaluate()