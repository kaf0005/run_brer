"""Unit tests and regression for PluginConfig classes."""
from run_brer.plugin_configs import TrainingPluginConfig, ConvergencePluginConfig, ProductionPluginConfig
from run_brer.run_data import GeneralParams, PairParams
import pytest

def test_plugins(raw_pair_data):
    """Build all three test plugins and check that they have all the required
    data to run.

    Parameters
    ----------
    raw_pair_data : dict
        A set of pair-specific parameters. Provided in conftest.py
    """
    tpc = TrainingPluginConfig()
    cpc = ConvergencePluginConfig()
    ppc = ProductionPluginConfig()

    gp = GeneralParams()
    general_parameter_defaults = gp.get_defaults()

    name = list(raw_pair_data.keys())[0]
    sites = raw_pair_data[name]["sites"]
    pp = PairParams(name)
    pp.load_sites(sites)
    #pp.set_to_defaults()

    pair_param_dict = pp.get_as_dictionary()

    tpc.scan_dictionary(general_parameter_defaults)
    pair_param_dict["phase"] = "training"
    tpc.scan_dictionary(pair_param_dict)
    # assert not tpc.get_missing_keys()

    cpc.scan_dictionary(general_parameter_defaults)
    pair_param_dict["phase"] = "convergence"
    cpc.scan_dictionary(pair_param_dict)
    # assert not cpc.get_missing_keys()

    ppc.scan_dictionary(general_parameter_defaults)
    pair_param_dict["phase"] = "production"
    ppc.scan_dictionary(pair_param_dict)
    # assert not ppc.get_missing_keys()

    with pytest.raises(KeyError):
        tpc.build_plugin()
    with pytest.raises(KeyError):
        cpc.build_plugin()
    with pytest.raises(KeyError):
        ppc.build_plugin()

    pp.set_to_defaults()
    tpc.scan_metadata(pp)
    cpc.scan_metadata(pp)
    ppc.scan_metadata(pp)
    
    tpc.build_plugin()
    cpc.build_plugin()
    ppc.build_plugin()
