import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

import fr_data_pr
import inception_v3
import inception

num_classes = 5

func = inception_v3
def network_fn(images):
    arg_scope = inception.inception_v3_arg_scope(weight_decay=weight_decay)
    with slim.arg_scope(arg_scope):
        return func(images, num_classes, is_training=is_training)


if hasattr(func, 'default_image_size'):
    network_fn.default_image_size = func.default_image_size

