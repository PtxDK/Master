import tensorflow as tf
import glob
import pathlib
import nibabel as nib


def load_img(val):
    def func(x,y):
        ret_x = nib.load(x.numpy().decode("utf-8")).get_fdata()[0]
        ret_y = nib.load(y.numpy().decode("utf-8")).get_fdata()[0]
        return ret_x, ret_y
    return func


def base_loader(base_dir, split_slices=True):
    base_dir = pathlib.Path(base_dir)
    images = (base_dir / "images").iterdir()
    labels = (base_dir / "labels").iterdir()
    vals = [[str(x),str(y)] for x,y in zip(images, labels)]
    dataset = tf.data.Dataset.from_tensor_slices(vals)
    # dataset = dataset.map(
    #     lambda x, y: tf.py_function(load_img(0), [x, y], [tf.float32, tf.int32])
    # )
    return dataset

