#!/usr/bin/env python
"""
Example run script for BRER simulations
"""

import run_brer.run_config as rc
import sys

sys.path.append('/home/jennifer/Git/brer/build/src/pythonmodule')

init = {
    'tpr': '/home/jennifer/Git/run_brer/run_brer/data/topol.tpr',
    'ensemble_dir': '/home/jennifer/test-brer',
    'ensemble_num': 5,
    'pairs_json': '/home/jennifer/Git/run_brer/run_brer/data/pair_data.json'
}

config = rc.RunConfig(**init)

"""config.run_data.set(A=100, name='196_228')"""

config.run()

"""self.pair_params[name].set('A',50)"""
