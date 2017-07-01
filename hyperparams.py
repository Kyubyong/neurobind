# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
June 2017 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/neurobind
'''

class Hyperparams:
    '''Hyperparameters'''
     
    # data paths
    train = 'uniprobe/CEH-22/CEH-22_deBruijn_v1.txt'  # or 'uniprobe/Oct-1/Oct-1_deBruijn_v1.txt'
    test = 'uniprobe/CEH-22/CEH-22_deBruijn_v2.txt'  # or 'uniprobe/Oct-1/Oct-1_deBruijn_v2.txt'
    
    # model
    hidden_units = 512  # alias = E
    num_blocks = 6  # number of encoder blocks
    dropout_rate = 0.2
    encoder_num_banks = 16
    num_highwaynet_blocks = 4
    norm_type = None # Either "bn",  "ln", "ins", or None
    loss_type = "l1" # Either "l1" or "l2"

    # training
    num_epochs = 20
    batch_size = 64  # alias = N
    lr = 0.0005  # learning rate.
    logdir = 'log'  # log directory
    results = "results" # results





