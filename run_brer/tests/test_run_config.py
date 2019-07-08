from run_brer.run_config import RunConfig
import pytest
import os


def test_run_config(tmpdir, data_dir, raw_pair_data):
    current_dir = os.getcwd()
    config_params = {
        "tpr": "{}/topol.tpr".format(data_dir),
        "ensemble_num": 1,
        "ensemble_dir": tmpdir,
        "pairs_json": "{}/pair_data.json".format(data_dir)
    }
    os.makedirs("{}/mem_{}".format(tmpdir, config_params["ensemble_num"]))
    rc = RunConfig(**config_params)
    rc.run_data.set(A=5, tau=0.1, tolerance=100, num_samples=2, sample_period=0.1, production_time=0.2)
    rc.run()
    rc.run()
    rc.run()
    os.chdir(current_dir)


# def test_converge(tmpdir, data_dir, raw_pair_data):
#     current_dir = os.getcwd()
#     config_params = {
#         "tpr": "{}/topol.tpr".format(data_dir),
#         "ensemble_num": 1,
#         "ensemble_dir": tmpdir,
#         "pairs_json": "{}/pair_data.json".format(data_dir)
#     }
#     os.makedirs("{}/mem_{}".format(tmpdir, config_params["ensemble_num"]))
#     rc = RunConfig(**config_params)
#     rc.run_data.set(A=5, tau=0.1, tolerance=100, num_samples=2, sample_period=0.1, production_time=0.2)
#     rc.run()
#     os.chdir(current_dir)
