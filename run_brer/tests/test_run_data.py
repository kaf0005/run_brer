"""Unit tests and regression for RunData classes."""

import pytest
from run_brer.run_data import GeneralParams, PairParams, RunData
from run_brer.pair_data import MultiPair, PairData


def test_general_parameters():
    """Set the defaults for general parameters and check no missing params."""
    gp = GeneralParams()
    gp.set_to_defaults()

    assert not gp.get_missing_keys()
    assert gp.get_as_dictionary() == gp.get_defaults()


def test_pair_parameters(raw_pair_data):
    """Set PairParams objects from PairData objects. Check that there are no
    missing keys.

    Parameters
    ----------
    raw_pair_data : dict
        raw pair data from conftest.py
    """
    for name in raw_pair_data:
        pd = PairData(name)
        pd.set_from_dictionary(raw_pair_data[name])
        pp = PairParams(name)
        pp.load_sites(sites=pd.get("sites"))

        assert pp.get_missing_keys() == ['alpha', 'target']
        pp.set_to_defaults()
        assert not pp.get_missing_keys()


def test_run_data(tmpdir, raw_pair_data):
    """Test creation of RunData object from PairData objects and check no
    missing keys.

    Parameters
    ----------
    tmpdir : str
        pytest temporary directory
    raw_pair_data : dict
        raw pair data from conftest.py
    """
    rd = RunData()

    for name in raw_pair_data:
        pd = PairData(name)
        pd.set_from_dictionary(raw_pair_data[name])
        rd.from_pair_data(pd)
        assert not rd.pair_params[name].get_missing_keys()

    assert not rd.general_params.get_missing_keys()

    # Test the setting function
    with pytest.raises(ValueError):
        rd.set(tau=0.1, name=name)
    with pytest.raises(ValueError):
        rd.set(alpha=1.)
    rd.set(alpha=1., name=name)
    
    # Test getting
    rd.get("alpha", name=name)
    with pytest.raises(ValueError):
        rd.get("alpha")

    # Test read/write of the state
    rd.save_config("{}/state.json".format(tmpdir))
    old_rd = rd
    rd = RunData()
    rd.load_config("{}/state.json".format(tmpdir))

    assert old_rd.as_dictionary() == rd.as_dictionary()

    # Test clearing pair data
    rd.clear_pair_data()
    assert rd.pair_params == {}