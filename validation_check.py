# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/neurobind.
'''

from __future__ import print_function

import os

from scipy.stats import spearmanr

from data_load import get_batch_data, load_data
from hyperparams import Hyperparams as hp
import tensorflow as tf
from train import Graph


def validation_check():
    # Load graph
    g = Graph(is_training=False); print("Graph loaded")

    # Load data
    X, Y = load_data(mode="val")

    with g.graph.as_default():
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir)); print("Restored!")

            # Get model
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1]  # model name

            # Inference
            if not os.path.exists(hp.results): os.mkdir(hp.results)
            with open(os.path.join(hp.results, "validation_results.txt"), 'a') as fout:
                expected, predicted = [], []
                for step in range(len(X) // hp.batch_size):
                    x = X[step * hp.batch_size: (step + 1) * hp.batch_size]
                    y = Y[step * hp.batch_size: (step + 1) * hp.batch_size]

                    # predict intensities
                    logits = sess.run(g.logits, {g.x: x})

                    expected.extend(list(y))
                    predicted.extend(list(logits))

                # Get spearman coefficients
                score, _ = spearmanr(expected, predicted)
                fout.write("{}\t{}\n".format(mname, score))

if __name__ == '__main__':
    validation_check()
    print("Done")

