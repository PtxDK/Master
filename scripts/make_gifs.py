#%%
import glob
from PIL import Image
#%%
images = glob.glob("../images/res-*.png")
images.sort()
ims = [Image.open(i) for i in images]
#%%
ims[0].save("../images/problems.gif",
            append_images=ims[1:],
            save_all=True,
            duration=500,
            version="GIF89a",
            loop=0)
