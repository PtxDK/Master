#%%
from heartnet.models import UNet2D
from heartnet.layers import UpConvBlock, ConvBlock

#%%
net = UNet2D(2, num_filters=64, depth=5, img_size=128)
net.create_model()