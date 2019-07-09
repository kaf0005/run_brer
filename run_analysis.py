#!/usr/bin/env python
"""
Example run script for BRER simulations
"""

import run_brer.mem_search as ms
import sys


init = {
    'ensemble_dir': '/home/kaf0005/UVA_reu/test-brer',
    'pairs_json': '/home/kaf0005/UVA_reu/analysis/pair_data.json',
    'analysis_dir': '/home/kaf0005/UVA_reu/analysis',
    'n':[1,2,3],
    'm':[0,1]
}



config = ms.Analysis(**init)
config.run()
