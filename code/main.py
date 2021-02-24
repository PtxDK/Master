#%%
from heartnet.models import UNet2D
from heartnet.layers.blocks import UpConvBlock, ConvBlock

#%%
net = UNet2D(2, 64, 5, 572)
net.create_model()