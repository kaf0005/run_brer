#!/usr/bin/env python
"""
Example run script for BRER simulations
"""

import run_brer.debug as rc
import sys


init = {
    'tpr': '/home/kaf0005/UVA_reu/analysis/syx.tpr',
    'ensemble_dir': '/home/kaf0005/UVA_reu/test-brer',
    'ensemble_num': 4,
    'pairs_json': '/home/kaf0005/UVA_reu/analysis/pair_data.json',
    'dict_json': 0
}

config = rc.Run(**init)
config.datDict()

