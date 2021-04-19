# Author: Michael Montalbano

# Contains functions for RNNs
import numpy as np


from hla_support import *
from metrics_binarized import *
import tensorflow as tf

def create_GRU(n_tokens, len_max):
    embed_size = 15
    model = keras.models.Sequential([
        keras.layers.Embedding(input_dim=n_tokens, output_dim=embed_size, input_length=len_max),
        keras.layers.GRU(15, return_sequences=True),
        keras.layers.GRU(15, return_sequences=False),
        keras.layers.Dense(1,activation="sigmoid")
    ])

    model.compile(loss="binary_crossentropy", optimizer="adam",
                metrics=[MyBinaryAccuracy(),
                          MyAUC()])

    return model


