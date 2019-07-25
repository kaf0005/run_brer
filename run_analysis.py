#!/usr/bin/env python
"""
Example analysis script for Correlation Structure simulations

To run this script, you must use the gmxapi software that includes gromacs command line operations
"""

import run_brer.mem_search as ms
import sys


init = {
    'tpr': '/home/kaf0005/UVA_reu/analysis/syx.tpr',
    'index_file': '/home/kaf0005/UVA_reu/analysis/beta_carbon_index.ndx',
    'select' : '25;26;27',
    'ensemble_dir': '/home/kaf0005/UVA_reu/test-brer',
    'pairs_json': '/home/kaf0005/UVA_reu/analysis/pair_data.json',
    'analysis_dir': '/home/kaf0005/UVA_reu/analysis',
    'n':[1,2,3],
    'm':[0,1]
}



config = ms.Analysis(**init)
config.run()
