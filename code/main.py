#%%
from heartnet.models import UNet2D

#%%
net = UNet2D(2, 64, 5, 572)
net.create_model()