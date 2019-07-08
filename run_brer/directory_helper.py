"""Organizes the directory structure for BRER runs. Creates directories on the
fly.

How the directory structure is organized:
    * This script should be run from your "top" directory (where
      you are planning to run all your ensemble members)
    * The top directory contains ensemble member subdirectories
      This script is intended to handle just ONE ensemble member,
      so we'll only be concerned with a single member subdirectory.
    * The example below is for the first iteration (iter 0) of one
      of the members. Future iterations would go in directories
      1,2,...y

"""

import os


class DirectoryHelper:
    def __init__(self, top_dir, param_dict):
        """Small class for manipulating a standard directory structure for BRER
        runs.

        Parameters
        ----------
        top_dir :
            the path to the directory containing all the ensemble members.
        param_dict :
            a dictionary specifying the ensemble number, the iteration,
            and the phase of the simulation.

        Examples
        --------
        >>> .
        >>> ├── 0
        >>> │   ├── converge_dist
        >>> │   │   ├── state.cpt
        >>> │   │   ├── state_prev.cpt
        >>> │   │   └── traj_comp.part0001.xtc
        >>> │   ├── production
        >>> │   │   ├── confout.part0005.gro
        >>> │   │   ├── state.cpt
        >>> │   │   ├── state_prev.cpt
        >>> │   │   ├── state_step4622560.cpt
        >>> │   │   ├── traj_comp.part0002.xtc
        >>> │   └── training
        >>> │       ├── state.cpt
        >>> │       ├── state_prev.cpt
        >>> │       └── traj_comp.part0001.xtc
        >>> ├── state.json
        >>> ├── submit.slurm
        >>> └── syx.tpr

        """

        self._top_dir = top_dir
        self._required_parameters = ['ensemble_num', 'iteration', 'phase']
        for required in self._required_parameters:
            if required not in param_dict:
                raise ValueError('Must define {}'.format(required))
        self._param_dict = param_dict

    def get_dir(self, level):
        """Get the directory for however far you want to go down the directory
        tree.

        Parameters
        ----------
        level :
            one of 'top', 'ensemble_num', 'iteration', or 'phase'.
            See the directory structure example provided at the beginning of this class.

        Returns
        -------
        type
            the path to the specified directory 'level' as a str.
        """
        pdict = self._param_dict
        if level == 'top':
            return_dir = self._top_dir
        elif level == 'ensemble_num':
            return_dir = '{}/mem_{}'.format(self._top_dir, pdict['ensemble_num'])
        elif level == 'iteration':
            return_dir = '{}/mem_{}/{}'.format(self._top_dir, pdict['ensemble_num'], pdict['iteration'])
        elif level == 'phase':
            return_dir = '{}/mem_{}/{}/{}'.format(self._top_dir, pdict['ensemble_num'], pdict['iteration'],
                                                  pdict['phase'])
        else:
            raise ValueError('{} is not a valid directory type for BRER simulations'.format('type'))
        return return_dir

    def build_working_dir(self):
        """Checks to see if the working directory for current state of BRER
        simulation exists. If it does not, creates the directory.
        """
        if not os.path.exists(self.get_dir('phase')):
            tree = [self.get_dir('ensemble_num'), self.get_dir('iteration')]
            for leaf in tree:
                if not os.path.exists(leaf):
                    os.mkdir(leaf)
            os.mkdir(self.get_dir('phase'))

    def change_dir(self, level):
        """Change to directory specified by level.

        Parameters
        ----------
        level : str
            How far to go down the directory tree.
            Can be one of 'top', 'ensemble_num', 'iteration', or 'phase'.
        """
        os.chdir(self.get_dir(level))
