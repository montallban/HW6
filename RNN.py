# Author: Michael Montalbano

# Contains functions for RNNs
import numpy as np


from hla_support import *
from metrics_binarized import *
import tensorflow as tf

# GRU network
# Consists of 1 embedding layer, 2 GRU layers, and 1 dense layer
# Each GRU layer allows l2regularization and dropout
# There are two branches, in case dropout or l2_reg == None (not sure if necessary but I'm paranoid)
def create_GRU(n_tokens, len_max, dropout, l2):
    embed_size = 15
    if dropout == None:
       model = keras.models.Sequential([
            keras.layers.Embedding(input_dim=n_tokens, output_dim=embed_size, input_length=len_max),
            keras.layers.GRU(15, return_sequences=True, kernel_regularizer=keras.regularizers.l2(l2), recurrent_regularizer=keras.regularizers.l2(l2)),
            keras.layers.GRU(15, return_sequences=False, kernel_regularizer=keras.regularizers.l2(l2), recurrent_regularizer=keras.regularizers.l2(l2)),
            keras.layers.Dense(1,activation="sigmoid")
        ]) 
    else:
        model = keras.models.Sequential([
            keras.layers.Embedding(input_dim=n_tokens, output_dim=embed_size, input_length=len_max),
            keras.layers.GRU(15, return_sequences=True, dropout=dropout, recurrent_dropout=dropout, 
                                kernel_regularizer=keras.regularizers.l2(l2),
                                recurrent_regularizer=keras.regularizers.l2(l2)),
            keras.layers.GRU(15, return_sequences=False, dropout=dropout, recurrent_dropout=dropout, 
                                kernel_regularizer=keras.regularizers.l2(l2), 
                                recurrent_regularizer=keras.regularizers.l2(l2)
                ),
            keras.layers.Dense(1,activation="sigmoid")
        ])

    model.compile(loss="mse", optimizer="adam",
                metrics=[MyBinaryAccuracy(),
                          MyAUC()])

    return model


