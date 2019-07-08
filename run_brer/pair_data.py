"""Classes to handle 1) pair metadata 2) resampling from DEER distributions at
each new BRER iteration."""

import numpy as np
from run_brer.metadata import MetaData, MultiMetaData
import json


class PairData(MetaData):
    """Class to handle pair metadata (distribution, bins, atom ids)"""

    def __init__(self, name):
        super().__init__(name=name)
        self.set_requirements(['distribution', 'bins', 'sites'])


class MultiPair(MultiMetaData):
    """Single class for handling multiple pair data.

    Handles resampling of targets.
    """

    def __init__(self):
        super().__init__()

    def read_from_json(self, filename='state.json'):
        """Reads pair data from json file. For an example file, see
        pair_data.json in the data directory.

        Parameters
        ----------
        filename : str, optional
            filename of the pair data, by default 'state.json'
        """
        self._metadata_list = []
        self._names = []
        data = json.load(open(filename, 'r'))
        for name, metadata in data.items():
            self._names.append(name)
            metadata_obj = PairData(name=name)
            metadata_obj.set_from_dictionary(metadata)
            self._metadata_list.append(metadata_obj)

    def re_sample(self):
        """Re-sample from the joint space. Do normalization just in case the
        data aren't normalized already.

        Returns
        -------
        dict
            dictionary of targets, drawn from DEER distributions.
        """
        answer = {}
        for pair_data in self._metadata_list:
            name = pair_data.__getattribute__('name')
            distribution = pair_data.get('distribution')
            bins = pair_data.get('bins')

            normalized = np.divide(distribution, np.sum(distribution))
            answer[name] = np.random.choice(bins, p=normalized)

        return answer
