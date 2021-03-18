from heartnet.config.base import YamlConfig
import tensorflow as tf
import pathlib
import nibabel as nib
from .preprocess import *


def base_loader(base_dir):
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

    def load_img(x, y):
        ret_x = nib.load(x.numpy().decode("utf-8")).get_fdata()
        ret_y = nib.load(y.numpy().decode("utf-8")).get_fdata()
        return ret_x, ret_y

    def get_img(x, y):
        return tf.py_function(load_img, [x, y], [tf.float32, tf.int32])

    dataset = dataset.map(
        get_img, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    return dataset


def load3D(base_dir, output_shape):
    dataset = base_loader(base_dir, False)
    dataset = dataset.map(crop_image_to_shape(output_shape))
    offset = (111-output_shape) // 2
    dataset = dataset.map(crop_slices(offset, output_shape))
    dataset = dataset.map(expand_dims)
    return dataset


def load2D(base_dir):
    dataset = base_loader(base_dir)
    dataset = dataset.map(reshape_slices)
    dataset = dataset.map(expand_dims)
    return dataset.flat_map(split_slices)


load_functions = {"UNet": load2D, "UNet3D": load3D}


def load_datasets(config: YamlConfig):
    load_function = load_functions[config["model"]]
    ret = {i: None for i in config.splits}
    base_folder = pathlib.Path(config["data"]["base_folder"])
    for split in config.splits:
        ret[split] = load_function(base_folder / split
                                  ).batch(config["fit"]["batch_size"])
    return ret