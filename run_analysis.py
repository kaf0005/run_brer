#!/usr/bin/env python
"""
Example run script for BRER simulations
"""

import run_brer.mem_search as rc
import sys


init = {
    'ensemble_dir': '/home/kaf0005/KF2019REU/jmh_dc3_runs',
    'pairs_json': '/home/kaf0005/KF2019REU/syntaxin_files/pair_data.json'
    'analysis_dir':'/home/kaf0005/KF2019REU/analysis'
}


config = rc.RunConfig(**init)
config.run()

