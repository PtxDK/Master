import tensorflow as tf
import pathlib
import nibabel as nib
import numpy as np


def load_img(x, y):

    ret_x = nib.load(x.numpy().decode("utf-8")).get_fdata()
    ret_y = nib.load(y.numpy().decode("utf-8")).get_fdata()
    ret_x = np.array([ret_x[:, :, i] for i in range(111)])
    ret_y = np.array([ret_y[:, :, i] for i in range(111)])
    return ret_x, ret_y


def base_loader(base_dir, split_slices=True):
    base_dir = pathlib.Path(base_dir)
    images = (base_dir / "images")
    labels = (base_dir / "labels")
    img_ds = tf.data.Dataset.from_tensor_slices(
        [str(i) for i in images.glob("*")]
    )
    label_ds = tf.data.Dataset.from_tensor_slices(
        [str(i) for i in labels.glob("*")]
    )
    dataset = tf.data.Dataset.zip((img_ds, label_ds))
    dataset = dataset.map(
        lambda x, y: tf.py_function(load_img, [x, y], [tf.float32, tf.int32]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    return dataset


def load3D(base_dir):
    return base_loader(base_dir)

def load2D(base_dir):
    dataset = base_loader(base_dir)
    return dataset.flat_map(
        lambda x, y: tf.data.Dataset.
        from_tensor_slices((x[:, :, :, None], y[:, :, :, None]))
    )