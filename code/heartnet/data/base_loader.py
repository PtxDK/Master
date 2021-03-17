import tensorflow as tf
import pathlib
import nibabel as nib
import numpy as np


def load_img(x, y):
    ret_x = nib.load(x.numpy().decode("utf-8")).get_fdata()
    ret_y = nib.load(y.numpy().decode("utf-8")).get_fdata()
    return ret_x, ret_y


def split_slices(x, y):
    return tf.transpose(x, [2, 0, 1]), tf.transpose(y, [2, 0, 1])


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

    def get_img(x, y):
        return tf.py_function(load_img, [x, y], [tf.float32, tf.int32])

    dataset = dataset.map(
        get_img, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    return dataset


def crop_image(o):

    def func(img, labels):
        return tf.image.resize_with_crop_or_pad(
            img, o, o
        ), tf.image.resize_with_crop_or_pad(labels, o, o)

    return func


def load3D(base_dir, output_shape):
    dataset = base_loader(base_dir, False)
    dataset = dataset.map(crop_image(output_shape))
    dataset = dataset.map(
        lambda x, y: (tf.expand_dims(x, -1), tf.expand_dims(y, -1))
    )
    return dataset


def load2D(base_dir):
    dataset = base_loader(base_dir)
    dataset = dataset.map(split_slices)
    return dataset.flat_map(
        lambda x, y: tf.data.Dataset.
        from_tensor_slices((x[:, :, :, None], y[:, :, :, None]))
    )