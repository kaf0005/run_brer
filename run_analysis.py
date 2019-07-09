#!/usr/bin/env python
"""
Example run script for BRER simulations
"""

import run_brer.mem_search as ms
import sys


init = {
    'ensemble_dir': '/home/kaf0005/KF2019REU/jmh_dc3_runs',
    'pairs_json': '/home/kaf0005/KF2019REU/syntaxin_files/pair_data.json',
    'analysis_dir': '/home/kaf0005/KF2019REU/analysis',
    'n':[1,2,3,4],
    'm':[1,2]
        }


config = ms.Analysis(**init)
config.run()
