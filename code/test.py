# from numpy.lib.npyio import load
import h5py
# from tensorflow.keras.models import load_model
model = h5py.File("./model/UNet3D_augmentation-shake-0.3330.h5")
print(model.keys())

