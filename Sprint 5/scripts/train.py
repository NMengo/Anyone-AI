"""
This script will be used for training our CNN. The only input argument it
should receive is the path to our configuration file in which we define all
the experiment settings like dataset, model output folder, epochs,
learning rate, data augmentation, etc.
"""
import argparse

import tensorflow as tf
from tensorflow import keras
from models import   resnet_50
from utils import utils
from keras.callbacks import ReduceLROnPlateau
import math
import os

# Prevent tensorflow to allocate the entire GPU
# https://www.tensorflow.org/api_docs/python/tf/config/experimental/set_memory_growth
physical_devices = tf.config.list_physical_devices("GPU")
for gpu in physical_devices:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass

def scheduler(epoch, lr):
    if epoch < 7: 
        return 0.001
    if epoch < 8:
        return 0.0005
    if epoch < 16:
        return 0.00005
    else:
        return 0.00001

# Supported optimizer algorithms
OPTIMIZERS = {
    "adam": keras.optimizers.Adam,
    "sgd": keras.optimizers.SGD,
    "rms": keras.optimizers.RMSprop,
}

# Supported callbacks
CALLBACKS = {
    "model_checkpoint": keras.callbacks.ModelCheckpoint,
    "tensor_board": keras.callbacks.TensorBoard,
    "early_stopping": keras.callbacks.EarlyStopping,
    "reduce_lr_plat": keras.callbacks.ReduceLROnPlateau,
}


def parse_args():
    """
    Use argparse to get the input parameters for training the model.
    """
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument(
        "warmup_config",
        type=str,
        help="Full path to WarmUp configuration file.",
    )
    parser.add_argument(
        "finetun_config",
        type=str,
        help="Full path to Fine-tuning configuration file.",
    )

    args = parser.parse_args()

    return args


def parse_optimizer(config):
    """
    Get experiment settings for optimizer algorithm.

    Parameters
    ----------
    config : str
        Experiment settings.
    """
    opt_name, opt_params = list(config["compile"]["optimizer"].items())[0]
    if opt_params is None:
        optimizer = OPTIMIZERS[opt_name]
    else:
        optimizer = OPTIMIZERS[opt_name](**opt_params)

    del config["compile"]["optimizer"]

    return optimizer


def parse_callbacks(config):
    """
    Add Keras callbacks based on experiment settings.

    Parameters
    ----------
    config : str
        Experiment settings.
    """
    callbacks = []
    if "callbacks" in config["fit"]:
        for callbk_name, callbk_params in config["fit"]["callbacks"].items():
            callbacks.append(CALLBACKS[callbk_name](**callbk_params))
            callbacks.append(keras.callbacks.LearningRateScheduler(scheduler))

        del config["fit"]["callbacks"]

    return callbacks


def main(warmup_config, finetun_config):
    """
    Code for the training logic.

    Parameters
    ----------
    config_file : str
        Full path to experiment configuration file.
    """
    # Load configuration file, use utils.load_config()
    warmup_config = utils.load_config(warmup_config)
    finetun_config = utils.load_config(finetun_config)

    # Get the list of output classes
    # We will use it to control the order of the output predictions from
    # keras is consistent
    class_names = utils.get_class_names(warmup_config)

    # Check if number of classes is correct
    if len(class_names) != warmup_config["model"]["classes"]:
        raise ValueError(
            "The number classes between your dataset and your model"
            "doen't match."
        )

    # Load training dataset
    # We will split train data in train/validation while training our
    # model, keeping away from our experiments the testing dataset
    train_ds = keras.preprocessing.image_dataset_from_directory(
        subset="training",
        class_names=class_names,
        seed=warmup_config["seed"],
        **warmup_config["data"],
    )
    val_ds = keras.preprocessing.image_dataset_from_directory(
        subset="validation",
        class_names=class_names,
        seed=warmup_config["seed"],
        **warmup_config["data"],
    )

    # Creates a Resnet50 model for finetuning
    cnn_model = resnet_50.create_model(**warmup_config["model"])
    print(cnn_model.summary())

    # Compile model, prepare for training
    optimizer = parse_optimizer(warmup_config)
    cnn_model.compile(
        optimizer=optimizer,
        **warmup_config["compile"],
    )

    if not os.path.exists('models/init_fcl_weights.h5'):
    # Start training!
        callbacks = parse_callbacks(warmup_config)
        cnn_model.fit(
            train_ds, validation_data=val_ds, callbacks=callbacks, **warmup_config["fit"]
        )
        print('=================================================')
        print('#1 Saving FCL Weights...')
        cnn_model.save_weights('models/init_fcl_weights.h5')
    else:
        print('=================================================')
        print('#1 Loading pre-trained FCL Weights...')
        cnn_model.load_weights('models/init_fcl_weights.h5')

    print('=================================================')
    print('#2 FCLayers already warmed up')
    print('#3 Unfreezing CONV Layers')
    base_model = cnn_model.get_layer('resnet50')
    base_model.trainable = True
    for layer in base_model.layers[:19]: #87
        layer.trainable = False       

    print('=================================================')
    print(cnn_model.summary())
    for i, layer in enumerate(base_model.layers):
        print(i, layer.name, '-', layer.trainable)

    print('#4 Recompiling the model')
    optimizer = parse_optimizer(finetun_config)
    cnn_model.compile(
        optimizer=optimizer,
        **finetun_config["compile"],
    )

    print('#5 Retraining both CONV and FC Layers')
    print('=================================================')
    callbacks = parse_callbacks(finetun_config)
    cnn_model.fit(
        train_ds, validation_data=val_ds, callbacks=callbacks, **finetun_config["fit"]
    )

    print('#6 Saving model...')
    print('=================================================')
    cnn_model.save('models/TLRN50_Fmodel.h5')
    print('#7 Saving weights...')
    print('=================================================')
    cnn_model.save_weights('models/TLRN50_Fmodel_weights10.h5')

if __name__ == "__main__":
    args = parse_args()
    main(args.warmup_config, args.finetun_config)
