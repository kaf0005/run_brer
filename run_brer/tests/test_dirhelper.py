"""Unit tests and regression for DirectoryHelper class."""
from run_brer.directory_helper import DirectoryHelper
import os


def test_directory(tmpdir):
    """Checks that directory creation for BRER runs is functional.

    Parameters
    ----------
    tmpdir :
        temporary pytest directory.
    """
    my_home = os.path.abspath(os.getcwd())
    top_dir = tmpdir.mkdir("top_directory")
    dir_helper = DirectoryHelper(top_dir, {'ensemble_num': 1, 'iteration': 0, 'phase': 'training'})
    dir_helper.build_working_dir()
    dir_helper.change_dir('phase')
    assert (os.getcwd() == '{}/mem_{}/{}/{}'.format(top_dir, 1, 0, 'training'))

    os.chdir(my_home)
