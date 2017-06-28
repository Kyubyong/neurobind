# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
June 2017 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/find_motifs
'''
from __future__ import print_function
from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import re

def load_vocab():
    vocab = "ACGT"
    nucl2idx = {nucl: idx for idx, nucl in enumerate(vocab)}
    idx2nucl = {idx: nucl for idx, nucl in enumerate(vocab)}
    return nucl2idx, idx2nucl

def load_data(mode="train"):
    nucl2idx, idx2nucl = load_vocab()

    def to_idx(probe):
        return [nucl2idx[nucl] for nucl in probe.strip()]

    xs, ys = [], []

    f = hp.train if mode in ("train", "val") else hp.test
    for line in open(f, "r").read().splitlines()[1:]:
        probe, intensity = line.split("\t")
        try:
            x = to_idx(line.split("\t")[0])
            y = float(intensity.split(".")[0])
        except:
            continue
        xs.append(x)
        ys.append(y)

    X = np.array(xs, np.int32)
    Y = np.array(ys, np.float32)

    if mode == "test":
        return X, Y
    elif mode == "train":
        return X[:int(len(X)*.7)], Y[:int(len(X)*.7)]
    elif mode == "val":
        return X[int(len(X)*.7):], Y[int(len(X)*.7):]
    else:
        raise ValueError("Mode must either `train`, `val`, or `test`.")

def get_batch_data():
    # Load data
    X, Y = load_data()

    # calc total batch count
    num_batch = len(X) // hp.batch_size

    # Convert to tensor
    X = tf.convert_to_tensor(X, tf.int32)
    Y = tf.convert_to_tensor(Y, tf.float32)

    # Create Queues
    input_queues = tf.train.slice_input_producer([X, Y])

    # create batch queues
    x, y = tf.train.batch(input_queues,
                          num_threads=8,
                          batch_size=hp.batch_size,
                          capacity=hp.batch_size * 64,
                          allow_smaller_final_batch=False)

    return x, y, num_batch  # (N, T), (N, T), ()

