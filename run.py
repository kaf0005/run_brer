#!/usr/bin/env python
"""
Example run script for BRER simulations
"""

import run_brer.run_config as rc


init = {
    'tpr': '/home/kaf0005/UVA_reu/analysis/syx.tpr',
    'ensemble_dir': '/home/kaf0005/UVA_reu/',
    'ensemble_num': 1,
    'pairs_json': '/home/kaf0005/UVA_reu/analysis/pair_data.json',
    'dict_json': 0
}

config = rc.RunConfig(**init)
config.run_data.set(max_train_time=10,tau=10)
for name in config.pairs.names:
    config.run_data.set(A=200,name=name)
config.run()

