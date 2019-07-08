"""Unit tests and regression for PairData classes."""
from run_brer.pair_data import PairData, MultiPair
import pytest


def test_pair_data(data_dir, raw_pair_data):
    """Ensures that multipair constructs multiple PairData objects.
    
    Parameters
    ----------
    data_dir : str
        pytest data directory
    raw_pair_data : dict
        dictionary of multiple pair data
    """
    mp = MultiPair()
    mp.read_from_json("{}/pair_data.json".format(data_dir))

    assert mp.get_as_single_dataset() == raw_pair_data
    for name in mp.names:
        assert (type(
            mp[mp.name_to_id(name)]) == PairData)

    samples = mp.re_sample()
    assert list(samples.keys()) == mp.names
    

# def test_pair_data(multi_pair_data, raw_pair_data):
#     """Ensures that multipair constructs multiple PairData objects.

#     Parameters
#     ----------
#     multi_pair_data : [type]
#         [description]
#     raw_pair_data : [type]
#         [description]
#     """
#     assert (multi_pair_data.get_as_single_dataset() == raw_pair_data)
#     for name in multi_pair_data.get_names():
#         assert (type(
#             multi_pair_data[multi_pair_data.name_to_id(name)]) == PairData)


# def test_resampling(multi_pair_data):
#     print(multi_pair_data.re_sample())
