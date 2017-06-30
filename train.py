# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
June 2017 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/neurobind
'''
from __future__ import print_function

from tqdm import tqdm

from data_load import get_batch_data, load_vocab, load_data
from hyperparams import Hyperparams as hp
from modules import *
import tensorflow as tf


class Graph:
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Load data
            if is_training:
                self.x, self.y, self.num_batch = get_batch_data()  # (N, T)
            else:
                self.x = tf.placeholder(tf.int32, shape=(None, 60))

            # Load vocabulary
            nucl2idx, idx2nucl = load_vocab()

            # Encoder
            ## Embedding
            enc = embedding(self.x,
                            zero_pad=False,
                             vocab_size=len(nucl2idx),
                             num_units=hp.hidden_units,
                             scale=False,
                             scope="enc_embed")

            # Encoder pre-net
            prenet_out = prenet(enc,
                                num_units=[hp.hidden_units, hp.hidden_units//2],
                                dropout_rate=hp.dropout_rate,
                                is_training=is_training)  # (N, T, E/2)

            # Encoder CBHG
            ## Conv1D bank
            enc = conv1d_banks(prenet_out,
                               K=hp.encoder_num_banks,
                               num_units=hp.hidden_units//2,
                               norm_type="hp.norm_type",
                               is_training=is_training)  # (N, T, K * E / 2)

            # ### Max pooling
            # enc = tf.layers.max_pooling1d(enc, 2, 2, padding="same")  # (N, T, K * E / 2)

            ### Conv1D projections
            enc = conv1d(enc, hp.hidden_units//2, 3, scope="conv1d_1")  # (N, T, E/2)
            enc = normalize(enc, type="hp.norm_type", is_training=is_training, activation_fn=tf.nn.relu, scope="norm1")
            enc = conv1d(enc, hp.hidden_units//2, 3, scope="conv1d_2")  # (N, T, E/2)
            enc = normalize(enc, type="hp.norm_type", is_training=is_training, activation_fn=tf.nn.relu, scope="norm2")
            enc += prenet_out  # (N, T, E/2) # residual connections

            ### Highway Nets
            for i in range(hp.num_highwaynet_blocks):
                enc = highwaynet(enc, num_units=hp.hidden_units//2,
                                 scope='highwaynet_{}'.format(i))  # (N, T, E/2)

            # Final linear projection
            _, T, E = enc.get_shape().as_list()
            enc = tf.reshape(enc, (-1, T*E))
            self.logits = tf.squeeze(tf.layers.dense(enc, 1))

            if is_training:
                # Loss
                if hp.loss_type == "l1":
                    self.loss = tf.reduce_mean(tf.abs(self.logits - self.y))
                else: # l2
                    self.loss = tf.reduce_mean(tf.squared_difference(self.logits, self.y))

                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr)
                self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

                # Summary
                tf.summary.scalar('loss', self.loss)
                tf.summary.merge_all()

if __name__ == '__main__':
    # Construct graph
    g = Graph()
    print("Graph loaded")

    # Load vocabulary
    nucl2idx, idx2nucl = load_vocab()

    # Start a session
    with g.graph.as_default():
        sv = tf.train.Supervisor(graph=g.graph,
                                 logdir=hp.logdir,
                                 save_model_secs=0)
        with sv.managed_session() as sess:
            for epoch in range(1, hp.num_epochs + 1):
                if sv.should_stop(): break
                for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                    sess.run(g.train_op)

                # Save
                gs = sess.run(g.global_step)
                sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))

    print("Done")


