from tensorflow.keras import Model
from mpunet.models import UNet
class MPUnet(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unet = UNet()
    def call():
        pass