from re import sub
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def camel_case(s):
  s = sub(r"(_|-)+", " ", s).title().replace(" ", "")
  return ''.join([s[0].upper(), s[1:]])

def create_data_aug_layer(data_aug_layer):
    """
    Parameters
    ----------
    data_aug_layer : dict
        Data augmentation settings coming from the experiment YAML config
        file.

    Returns
    -------
    data_augmentation : keras.Sequential
        Sequential model having the data augmentation layers inside.
    """
    data_aug_layers = list(eval('layers.'+f"{camel_case(key)}(**{data_aug_layer[key]})") 
        for key in data_aug_layer.keys())
    data_augmentation = keras.Sequential(data_aug_layers)

    return data_augmentation
