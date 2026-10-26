# -*- coding: utf-8 -*-
"""
Created on Mon May 10 10:17:09 2021

@author: mmolina
"""

import os
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Folder of this file (ACME/python): paths below do not depend on the working directory
__C.base_dir = os.path.dirname(os.path.abspath(__file__))

#########################
#      Data Loader      #
#########################
__C.dim=3 # For 3D captures
__C.norm = None # one of None, 'instance_norm', 'batch_norm'
__C.patch_size_3D = [256, 256, 16]
__C.patch_size_2D = [256, 256]
__C.patch_size = __C.patch_size_2D if __C.dim == 2 else __C.patch_size_3D

# number of classes for head classifier network: cell + 1 (background)
__C.head_classes = 2
# number of classes for the full problem network: venule, cell + 1 (background)
__C.num_seg_classes = 3 

#########################
#     Model options     #
#########################
__C.weight_init = None
__C.relu = 'relu'
__C.pool_size = (7, 7) if __C.dim == 2 else (7, 7, 3)
__C.start_filts = 48 if __C.dim == 2 else 18
__C.end_filts = __C.start_filts * 4 if __C.dim == 2 else __C.start_filts * 2
__C.pyramid_levels = [0, 1, 2]
__C.model_min_confidence = 0.1

#########################
#      Test options     #
#########################

__C.batch_size_3d=1 # Only batchsize==1 supported
__C.depth=16

#########################
#  Experiment options   #
#########################

__C.stats_dir = os.path.abspath(os.path.join(__C.base_dir, '..', 'data', 'statistics'))
__C.data_dir = os.path.abspath(os.path.join(__C.base_dir, '..', 'database'))
__C.model_dir = os.path.join(__C.base_dir, 'models')
__C.model_name='ACME'

# Load statistics
__C.mean=np.load(os.path.join(__C.stats_dir,'mean-frame.npy'))
__C.std=np.load(os.path.join(__C.stats_dir,'std-frame.npy'))

#########################
#   Training options    #
#########################
# Used by detection_train.py. Data are read from data_dir/train (training) and
# data_dir/test (validation); results go to model_dir/TRAIN.model_name.

__C.TRAIN = edict()
__C.TRAIN.stage = 'both'                # 1: segmentation, 2: classifier fine-tuning, 'both': 1 then 2
__C.TRAIN.model_name = 'ACME_train'     # experiment name (subfolder of model_dir)
__C.TRAIN.init_weights = None           # initial checkpoint, e.g. os.path.join(__C.model_dir, 'ACME.pth').
                                        # Stage 2 alone: None -> best stage-1 checkpoint of this experiment
__C.TRAIN.stage2_from = 'acc'           # stage-1 checkpoint that starts stage 2: 'acc' (cell IoU) or 'auc'
__C.TRAIN.recompute_stats = False       # True: mean/std from the training set instead of stats_dir

# Stage 1: segmentation training (Adam, all parameters, StepLR)
__C.TRAIN.epochs1 = 24
__C.TRAIN.lr1 = 1e-3
__C.TRAIN.step1 = 8                     # StepLR step size (epochs)
__C.TRAIN.gamma1 = 0.1                  # StepLR decay factor

# Stage 2: classifier fine-tuning (Adam, differential learning rates, StepLR)
__C.TRAIN.epochs2 = 15
__C.TRAIN.lr2_body = 1e-5               # U-Net encoders, decoders and segmentation/venule/aux heads
__C.TRAIN.lr2_heads = 1e-4              # feature-pyramid convolutions and classifier
__C.TRAIN.step2 = 3
__C.TRAIN.gamma2 = 0.5

# Common
__C.TRAIN.weight_decay = 0.0
__C.TRAIN.num_workers = 0               # DataLoader workers
__C.TRAIN.save_every = 0                # also keep a checkpoint every N epochs (0: off)
__C.TRAIN.log_every = 10                # batches between log lines
__C.TRAIN.seed = 0

#########################
#  Data visualization   #
#########################

__C.POINT_SIZE=60
__C.ALPHA=1
__C.N_features=74
__C.alg='tsne' # 'tsne' or 'umap'
__C.graphs_number=5
if (__C.alg=='tsne'):
    # param1: perplexity and param2: exaggeration
    __C.param1=np.linspace(1,197,50)
    __C.param2=np.linspace(1,57,29)
else:
    # param1: number of neighbors and param2: minimum distance
    __C.param1=np.linspace(3,199,50)
    __C.param2=np.linspace(0.1,0.9,9)


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        if type(b[k]) is not type(v):
            raise ValueError(('Type mismatch ({} vs. {}) '
                              'for config key: {}').format(type(b[k]),
                                                           type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.safe_load(f))

    _merge_a_into_b(yaml_cfg, __C)

def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value
