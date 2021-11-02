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

__C.stats_dir = os.path.abspath("../data/statistics")
__C.data_dir = os.path.abspath("../database")
__C.model_dir = 'models'
__C.model_name='ACME'

# Load statistics
__C.mean=np.load(os.path.join(__C.stats_dir,'mean-frame.npy'))
__C.std=np.load(os.path.join(__C.stats_dir,'std-frame.npy'))

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

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
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
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)

def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert d.has_key(subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value
