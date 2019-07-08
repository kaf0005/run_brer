#!/usr/bin/env python
"""
Example run script for BRER simulations
"""

import run_brer.run_config as rc
import sys


init = {
    'tpr': '/home/kaf0005/KF2019REU/syntaxin_files/syx.tpr',
    'ensemble_dir': '/home/kaf0005/KF2019REU/jmh_dc3_runs',
    'ensemble_num': 15,
    'pairs_json': '/home/kaf0005/KF2019REU/syntaxin_files/pair_data.json'
}


"""init = {
    'tpr': '/home/jennifer/Git/run_brer/run_brer/data/topol.tpr',
    'ensemble_dir': '/home/jennifer/test-brer',
    'ensemble_num': 5,
    'pairs_json': '/home/jennifer/Git/run_brer/run_brer/data/pair_data.json'
}"""

config = rc.RunConfig(**init)

#config.run_data.set('A'=25,  name=('052_210')
#config.run_data.set('A'=25,  name=('105_216')
#config.run_data.set('A'=25,  name=('196_228')

config.run()

