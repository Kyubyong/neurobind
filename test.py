# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/neurobind.
'''

from __future__ import print_function

import os
import sys

from scipy.stats import spearmanr

from data_load import get_batch_data, load_vocab, load_data
from hyperparams import Hyperparams as hp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from train import Graph


def eval():
    # Load graph
    g = Graph(is_training=False); print("Graph loaded")

    # Load data
    X, Y = load_data(mode="test")
    nucl2idx, idx2nucl = load_vocab()

    with g.graph.as_default():
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir)); print("Restored!")

            # Get model name
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1]  # model name

            # Inference
            if not os.path.exists(hp.results): os.mkdir(hp.results)
            with open(os.path.join(hp.results, mname), 'w') as fout:
                fout.write("{}\t{}\t{}\n".format("probe", "expected intensity", "predicted intensity"))
                expected, predicted = [], []
                for step in range(len(X) // hp.batch_size):
                    x = X[step * hp.batch_size: (step + 1) * hp.batch_size]
                    y = Y[step * hp.batch_size: (step + 1) * hp.batch_size]

                    # predict intensities
                    logits = sess.run(g.logits, {g.x: x})

                    expected.extend(list(y))
                    predicted.extend(list(logits))

                    for xx, yy, ll in zip(x, y, logits):  # sequence-wise
                        fout.write("{}\t{}\t{}\n".format("".join(idx2nucl[idx] for idx in xx), yy, ll))

                # Get spearman coefficients
                score, _ = spearmanr(expected, predicted)
                fout.write("{}{}\n".format("Spearman Coefficient: ", score))

                # Plot the ranks of the top 100 positive probes
                expected_predicted = sorted(zip(expected, predicted), key=lambda x: float(x[0]), reverse=True)
                expected_predicted = [list(each) + [int(i < 100)] for i, each in enumerate(expected_predicted)]
                expected_predicted = sorted(expected_predicted, key=lambda x: float(x[1]), reverse=True)
                predicted_ranks = np.array([each[-1] for each in expected_predicted])

                # Plot
                axprops = dict(xticks=[], yticks=[])
                barprops = dict(aspect='auto', cmap=plt.cm.binary, interpolation='nearest')
                
                fig = plt.figure()
                
                predicted_ranks.shape = len(predicted_ranks), 1
                ax = fig.add_axes([0, 0, .5, 1], **axprops)
                ax.imshow(predicted_ranks, **barprops)
                fig.savefig('fig/rank.png')

if __name__ == '__main__':
    eval()
    print("Done")

