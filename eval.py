# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/find_motifs.
Usage:
`python eval.py val` for validation check
`python eval.py test` for test results
'''

from __future__ import print_function

import os

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from train import Graph
from data_load import get_batch_data, load_vocab, load_data
from scipy.stats import spearmanr
import sys

def eval(mode):
    '''
    Get a Spearman rank-order correlation coefficient.

    Args:
      mode: A string. Either `val` or `test`.
    '''
    # Set save directory
    savedir = hp.valdir if mode=="val" else hp.testdir

    # Load graph
    g = Graph(is_training=False)
    print("Graph loaded")

    # Load data
    X, Y = load_data(mode=mode)
    nucl2idx, idx2nucl = load_vocab()

    with g.graph.as_default():
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")

            # Get model
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1]  # model name

            # Inference
            if not os.path.exists(savedir): os.mkdir(savedir)
            with open("{}/{}".format(savedir, mname), 'w') as fout:
                fout.write("{}\t{}\t{}]\n".format("probe", "expected intensity", "predicted intensity"))
                expected, got = [], []
                for step in range(len(X) // hp.batch_size):
                    x = X[step * hp.batch_size: (step + 1) * hp.batch_size]
                    y = Y[step * hp.batch_size: (step + 1) * hp.batch_size]

                    # predict nucl
                    logits = sess.run(g.logits, {g.x: x})

                    for xx, yy, ll in zip(x, y, logits):  # sequence-wise
                        fout.write("{}\t{}\t{}\n".format("".join(idx2nucl[idx] for idx in xx), yy, ll))
                        expected.append(yy)
                        got.append(ll)

                # Spearman rank coefficient
                score, _ =spearmanr(expected, got)
                fout.write("Spearman rank correlation coefficients: " + str(score))


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else "val"
    eval(mode=mode)
    print("Done")

