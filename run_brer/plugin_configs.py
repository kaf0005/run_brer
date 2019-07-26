"""Classes used to build gmxapi plugins for all phases of a BRER iteration Each
class corresponds to ONE restraint since gmxapi plugins each correspond to one
restraint."""

from run_brer.metadata import MetaData
from abc import abstractmethod
import gmx


class PluginConfig(MetaData):
    """Abstract class used to build training, convergence, and production
    plugins."""

    def __init__(self):
        super().__init__('build_plugin')

    def scan_dictionary(self, dictionary):
        """Scans a dictionary and stores whatever parameters it needs for the
        build_plugin.

        Parameters
        ----------
        dictionary : dict
            a dictionary containing metadata, some of which may be needed for the run.
            The dictionary may contain *extra* data, i.e., this can be a superset of the
            needed plugin data.
        """

        for requirement in self.get_requirements():
            if requirement in dictionary.keys():
                self._metadata[requirement] = dictionary[requirement]

    def scan_metadata(self, data):
        """This scans a RunData or PairData obj and stores whatever parameters
        it needs for a run.

        Parameters
        ----------
        data :
            either type RunData or type PairData
        """
        self.scan_dictionary(data.get_as_dictionary())

    # def set_parameters(self, **kwargs):
    #     """

    #     Parameters
    #     ----------
    #     **kwargs :

    #     Returns
    #     -------

    #     """
    #     self.scan_dictionary(kwargs)

    @abstractmethod
    def build_plugin(self):
        """Abstract method for building a plugin.

        To be determined by the phase of the simulation (training,
        convergence, production)
        """
        pass


class TrainingPluginConfig(PluginConfig):
    def __init__(self):
        super().__init__()
        self.name = 'training'
        self.set_requirements(['sites', 'target', 'A', 'tau','max_train_time', 'tolerance', 'num_samples', 'logging_filename'])

    def build_plugin(self):
        """Builds training phase plugin for BRER simulations.

        Returns
        -------
        WorkElement
            a gmxapi WorkElement to be added to the workflow graph

        Raises
        ------
        KeyError
            if required parameters for building the plugin are missing.
        """

        if self.get_missing_keys():
            raise KeyError('Must define {}'.format(self.get_missing_keys()))
        potential = gmx.workflow.WorkElement(namespace="myplugin",
                                             operation="brer_restraint",
                                             depends=[],
                                             params=self.get_as_dictionary())
        potential.name = '{}'.format(self.get('sites'))
        return potential


class ConvergencePluginConfig(PluginConfig):
    def __init__(self):
        super().__init__()
        self.name = 'convergence'
        self.set_requirements(['sites', 'alpha', 'target', 'tolerance', 'sample_period', 'logging_filename'])

    def build_plugin(self):
        """Builds convergence phase plugin for BRER simulations.

        Returns
        -------
        WorkElement
            a gmxapi WorkElement to be added to the workflow graph

        Raises
        ------
        KeyError
            if required parameters for building the plugin are missing.
        """
        if self.get_missing_keys():
            raise KeyError('Must define {}'.format(self.get_missing_keys()))
        potential = gmx.workflow.WorkElement(namespace="myplugin",
                                             operation="linearstop_restraint",
                                             depends=[],
                                             params=self.get_as_dictionary())
        potential.name = '{}'.format(self.get('sites'))
        return potential


class ProductionPluginConfig(PluginConfig):
    def __init__(self):
        super().__init__()
        self.name = 'production'
        self.set_requirements(['sites', 'target', 'alpha', 'sample_period', 'logging_filename'])

    def build_plugin(self):
        """Builds production phase plugin for BRER simulations.

        Returns
        -------
        WorkElement
            a gmxapi WorkElement to be added to the workflow graph

        Raises
        ------
        KeyError
            if required parameters for building the plugin are missing.
        """
        if self.get_missing_keys():
            raise KeyError('Must define {}'.format(self.get_missing_keys()))
        potential = gmx.workflow.WorkElement(namespace="myplugin",
                                             operation="linear_restraint",
                                             depends=[],
                                             params=self.get_as_dictionary())
        potential.name = '{}'.format(self.get('sites'))
        return potential
