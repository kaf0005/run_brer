#!/usr/bin/env python
"""
Example run script for BRER simulations
"""

import run_brer.run_config as rc
import sys


init = {
    'tpr': '/home/kaf0005/KF2019REU/syntaxin_files/syx.tpr',
    'ensemble_dir': '/home/kaf0005/KF2019REU/jmh_dc3_runs',
    'ensemble_num': 8,
    'pairs_json': '/home/kaf0005/KF2019REU/syntaxin_files/pair_data.json',
    'dict_json': 0
}

config = rc.RunConfig(**init)
config.run()

