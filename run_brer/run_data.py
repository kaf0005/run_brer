"""
Class that handles the simulation data for BRER simulations
<doi!>
"""
from run_brer.metadata import MetaData
from run_brer.pair_data import PairData
import json


class GeneralParams(MetaData):
    """Stores the parameters that are shared by all restraints in a single simulation.
    These include some of the "Voth" parameters: tau, A, tolerance - Kasey is messing around with A, making is a specific parameter :)

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(self):
        super().__init__('general')
        self.set_requirements([
            'ensemble_num', 'iteration', 'phase', 'start_time','tau',
            'tolerance', 'num_samples', 'sample_period', 'production_time'
        ])


class PairParams(MetaData):
    """Stores the parameters that are unique to a specific restraint."""

    def __init__(self, name):
        super().__init__(name)
        self.set_requirements(['sites', 'logging_filename', 'alpha', 'target','A'])


class RunData:
    """Stores (and manipulates, to a lesser extent) all the metadata for a BRER run."""

    def __init__(self):
        """
        The full set of metadata for a single BRER run include both the general parameters
        and the pair-specific parameters.
        """
        self.general_params = GeneralParams()
        self.__defaults_general = {
            'ensemble_num': 1,
            'iteration': 0,
            'phase': 'training',
            'start_time': 0,
            'tau': 50,
            'tolerance': 0.1,
            'num_samples': 50,
            'sample_period': 100,
            'production_time': 10000  # 10 ns
        }
        self.general_params.set_from_dictionary(self.__defaults_general)
        self.pair_params = {}
        self.__names = []





    def set(self, name=None, **kwargs):
        """method used to set either general or a pair-specific parameter.

        Parameters
        ----------
        name :
            restraint name. These are the same identifiers that are used in the RunConfig (Default value = None)
w        kwargs :
            parameters and their values.
        **kwargs :


        Returns
        -------

        """
        for key, value in kwargs.items():
            # If a restraint name is not specified, it is assumed that the parameter is a "general" parameter.
            if not name:
                if key in self.general_params.get_requirements():
                    self.general_params.set(key, value)
                else:
                    raise ValueError(
                        'You have provided a name; this means you are probably trying to set a '
                        'pair-specific parameter. {} is not pair-specific'.
                        format(key))
            else:
                if key in self.pair_params[name].get_requirements():
                    self.pair_params[name].set(key, value)
                else:
                    raise ValueError(
                        '{} is not a pair-specific parameter'.format(key))

    def get(self, key, name=None):
        """get either a general or a pair-specific parameter.

        Parameters
        ----------
        key :
            the parameter to get.
        name :
            if getting a pair-specific parameter, specify the restraint name. (Default value = None)

        Returns
        -------
        type
            the parameter value.

        """
        if key in self.general_params.get_requirements():
            return self.general_params.get(key)
        elif name:
            return self.pair_params[name].get(key)
        else:
            raise ValueError(
                'You have not provided a name, but are trying to get a pair-specific parameter. '
                'Please provide a pair name')

    def as_dictionary(self):
        """Get the run metadata as a heirarchical dictionary:
        ├── pair parameters
        │   ├── name of pair 1
        │   │   ├── alpha
        │   │   ├── target
        │   │   └── ...
        │   ├── name of pair 2
        |
        ├── general parameters
            ├── A
            ├── tau
            ├── ...

        :return: heirarchical dictionary of metadata

        Parameters
        ----------

        Returns
        -------

        """
        pair_param_dict = {}
        for name in self.pair_params.keys():
            pair_param_dict[name] = self.pair_params[name].get_as_dictionary()

        return {
            'general parameters': self.general_params.get_as_dictionary(),
            'pair parameters': pair_param_dict
        }

    def from_dictionary(self, data):
        """Loads metadata into the class from a dictionary.

        Parameters
        ----------
        data :
            RunData metadata as a dictionary.

        Returns
        -------

        """
        self.general_params.set_from_dictionary(data['general parameters'])
        for name in data['pair parameters'].keys():
            self.pair_params[name] = PairParams(name)
            self.pair_params[name].set_from_dictionary(
                data['pair parameters'][name])

    def from_pair_data(self, pd: PairData):
        """Load some of the run metadata from a PairData object.
        Useful at the beginning of a run.

        Parameters
        ----------
        pd :
            PairData object from which metadata are loaded
        pd: PairData :


        Returns
        -------

        """
        name = pd.name
        self.pair_params[name] = PairParams(name)
        self.pair_params[name].set('sites', pd.get('sites'))
        self.pair_params[name].set('logging_filename', '{}.log'.format(name))
        self.pair_params[name].set('alpha', 0)
        self.pair_params[name].set('target', 3.0)
        self.pair_params[name].set('A',50)

    def clear_pair_data(self):
        """ """
        self.pair_params = {}

    def save_config(self, fnm='state.json'):
        """

        Parameters
        ----------
        fnm :
             (Default value = 'state.json')

        Returns
        -------

        """
        json.dump(self.as_dictionary(), open(fnm, 'w'))

    def load_config(self, fnm='state.json'):
        """

        Parameters
        ----------
        fnm :
             (Default value = 'state.json')

        Returns
        -------

        """
        self.from_dictionary(json.load(open(fnm)))
