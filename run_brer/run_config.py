from run_brer.run_data import RunData
from run_brer.pair_data import MultiPair
from run_brer.plugin_configs import TrainingPluginConfig, ConvergencePluginConfig, ProductionPluginConfig
from run_brer.directory_helper import DirectoryHelper
from copy import deepcopy
import os
import shutil
import logging
import gmx
import json
import atexit

class RunConfig:
    """Run configuration for single BRER ensemble member."""


    def __init__(self,
                tpr,
                ensemble_dir,
                ensemble_num=1,
                pairs_json='pair_data.json',
                pair=[],
                data=[],
                A1=[],
                A_parameter=[],
                ):
        """
        The run configuration specifies the files and directory structure used for the run.
        It determines whether the run is in the training, convergence, or production phase,
        then performs the run.
        :param tpr: path to tpr. Must be gmx 2017 compatible.
        :param ensemble_dir: path to top directory which contains the full ensemble.
        :param ensemble_num: the ensemble member to run.
        :param pairs_json: path to file containing *ALL* the pair metadata. An example of
        what such a file should look like is provided in the examples directory.

        :
        """

        self.tpr = tpr
        self.ens_dir = ensemble_dir
        self.pair = pair
        self.data = data
        self.A1=A1
        self.A_parameter = 1


        # a list of identifiers of the residue-residue pairs that will be restrained
        self.__names = []

        # Load the pair data from a json. Use this to set up the run metadata
        self.pairs = MultiPair()
        self.pairs.read_from_json(pairs_json)
        # use the same identifiers for the pairs here as those provided in the pair metadata
        # file this prevents mixing up pair data amongst the different pairs (i.e.,
        # accidentally applying the restraints for pair 1 to pair 2.)
        self.__names = self.pairs.get_names()

        self.run_data = RunData()
        self.run_data.set(ensemble_num=ensemble_num)

        self.state_json = '{}/mem_{}/state.json'.format(
            ensemble_dir, self.run_data.get('ensemble_num'))
        # If we're in the middle of a run, load the BRER checkpoint file and continue from
        # the current state.
        if os.path.exists(self.state_json):
            self.run_data.from_dictionary(json.load(open(self.state_json)))
        # Otherwise, populate the state information using the pre-loaded pair data. Then save
        # the current state.
        else:
            for pd in self.pairs:
                self.run_data.from_pair_data(pd)
            self.run_data.save_config(self.state_json)

                # List of plugins
        self.__plugins = []

        # Logging
        self._logger = logging.getLogger('BRER')
        self._logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler('brer{}.log'.format(ensemble_num))
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        # create formatter and add it to the handlers
        formatter = logging.Formatter(
            '%(asctime)s:%(name)s:%(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        self._logger.addHandler(fh)
        self._logger.addHandler(ch)

        self._logger.info("Initialized the run configuration: {}".format(
            self.run_data.as_dictionary()))
        self._logger.info("Names of restraints: {}".format(self.__names))

        # Need to cleanly handle cancelled jobs: write out a checkpoint of the state if the
        # job is exited.
        def cleanup():
            """ """
            self.run_data.save_config(self.state_json)
            self._logger.info(
                "BRER received INT signal, stopping and saving data to {}".
                format(self.state_json))

        atexit.register(cleanup)

    def build_plugins(self, plugin_config):
        """

        Parameters
        ----------
        plugin_config :


        Returns
        -------

        """
        # One plugin per restraint.
        # TODO: what is the expected behavior when a list of plugins exists? Probably wipe them.

        self.__plugins = []
        general_params = self.run_data.as_dictionary()['general parameters']

        # For each pair-wise restraint, populate the plugin with data: both the "general" data and
        # the data unique to that restraint.
        for name in self.__names:
            pair_params = self.run_data.as_dictionary()['pair parameters'][
                name]
            new_restraint = deepcopy(plugin_config)
            new_restraint.scan_dictionary(
                general_params)  # load general data into current restraint
            new_restraint.scan_dictionary(
                pair_params)  # load pair-specific data into current restraint
            self.__plugins.append(new_restraint.build_plugin())

    def __change_directory(self):
        # change into the current working directory (ensemble_path/member_path/iteration/phase)
        dir_help = DirectoryHelper(
            top_dir=self.ens_dir,
            param_dict=self.run_data.general_params.get_as_dictionary())
        dir_help.build_working_dir()
        dir_help.change_dir('phase')

    def __move_cpt(self):
        current_iter = self.run_data.get('iteration')
        ens_num = self.run_data.get('ensemble_num')
        phase = self.run_data.get('phase')

        # If the cpt already exists, don't overwrite it
        if os.path.exists(
                '{}/mem_{}/{}/{}/state.cpt'.format(self.ens_dir, ens_num,
                                                   current_iter, phase)):
            self._logger.info(
                "Phase is {} and state.cpt already exists: not moving any files".
                format(phase))

        else:
            member_dir = '{}/mem_{}'.format(self.ens_dir, ens_num)
            prev_iter = current_iter - 1

            if phase in ['training', 'convergence']:
                if prev_iter > -1:
                    # Get the production cpt from previous iteration
                    gmx_cpt = '{}/{}/production/state.cpt'.format(
                        member_dir, prev_iter)
                    shutil.copy(gmx_cpt, '{}/state.cpt'.format(os.getcwd()))

                else:
                    pass  # Do nothing

            else:
                # Get the convergence cpt from current iteration
                gmx_cpt = '{}/{}/convergence/state.cpt'.format(
                    member_dir, current_iter)
                shutil.copy(gmx_cpt, '{}/state.cpt'.format(os.getcwd()))

    def __train(self):

        #do re-sampling
        targets = self.pairs.re_sample()
        self._logger.info('New targets: {}'.format(targets))
        for name in self.__names:
            self.run_data.set(name=name, target=targets[name])

        #save the new targets to the BRER checkpoint file.
        self.run_data.save_config(fnm=self.state_json)

        # backup existing checkpoint.
        # TODO: Don't backup the cpt, actually use it!!
        cpt = '{}/state.cpt'.format(os.getcwd())
        if os.path.exists(cpt):
            self._logger.warning(
            'There is a checkpoint file in your current working directory, but you are '
            'training. The cpt will be backed up and the run will start over with new targets'
            )
            if self.A_parameter==0:
                pass
            else:
                shutil.move(cpt, '{}.bak'.format(cpt))


        # If this is not the first BRER iteration, grab the checkpoint from the production
        # phase of the last round
            self.__move_cpt()

        # Set up a dictionary to go from plugin name -> restraint name
            sites_to_name = {}

        # Build the gmxapi session.
            md = gmx.workflow.from_tpr(self.tpr, append_output=False)
            self.build_plugins(TrainingPluginConfig())
            for plugin in self.__plugins:
                plugin_name = plugin.name
                for name in self.__names:
                    run_data_sites = "{}".format(
                        self.run_data.get('sites', name=name))
                    if run_data_sites == plugin_name:
                                sites_to_name[plugin_name] = name
                md.add_dependency(plugin)
            context = gmx.context.ParallelArrayContext(
                md, workdir_list=[os.getcwd()])

        # Run it.
            with context as session:
                session.run()

        # In the future runs (convergence, production) we need the ABSOLUTE VALUE of alpha.
            self._logger.info("=====TRAINING INFO======\n")

            for i in range(len(self.__names)):
                current_name = sites_to_name[context.potentials[i].name]
                self.data[i]=current_name
                current_alpha = context.potentials[i].alpha
                current_target = context.potentials[i].target

                self.run_data.set(name=current_name, alpha=current_alpha)
                self.run_data.set(name=current_name, target=current_target)
                self._logger.info("Plugin {}: alpha = {}, target = {}".format(
                    current_name,
                    current_alpha,
                    current_target)
            )

    def __converge(self):
        if self.A_parameter==0:
            pass
        else:
            self.__move_cpt()

        md = gmx.workflow.from_tpr(self.tpr, append_output=False)
        self.build_plugins(ConvergencePluginConfig())
        for plugin in self.__plugins:
            md.add_dependency(plugin)
        context = gmx.context.ParallelArrayContext(
            md, workdir_list=[os.getcwd()])
        with context as session:
            session.run()

	# Get the absolute time (in ps) at which the convergence run finished.
        # This value will be needed if a production run needs to be restarted.

        self.run_data.set(start_time=context.potentials[0].time)

        for i in range(len(self.__names)):
            self.A1    = self.run_data.get('A',name=self.pair[i])
            self.data = context.potentials[0].time
        


        self._logger.info("=====CONVERGENCE INFO======\n")
        for name in self.__names:
            current_alpha = self.run_data.get('alpha', name=name)
            current_target = self.run_data.get('target', name=name)
            self._logger.info("Plugin {}: alpha = {}, target = {}".format(
                name,
                current_alpha,
                current_target)
            )

    def __production(self):

        # Get the checkpoint file from the convergence phase
        self.__move_cpt()

        # Calculate the time (in ps) at which the BRER iteration should finish.
        # This should be: the end time of the convergence run + the amount of time for
        # production simulation (specified by the user).
        end_time = self.run_data.get('production_time') + self.run_data.get(
            'start_time')

        md = gmx.workflow.from_tpr(
            self.tpr, end_time=end_time, append_output=False)

        self.build_plugins(ProductionPluginConfig())
        for plugin in self.__plugins:
            md.add_dependency(plugin)
        context = gmx.context.ParallelArrayContext(
            md, workdir_list=[os.getcwd()])
        with context as session:
            session.run()

        self._logger.info("=====PRODUCTION INFO======\n")
        for name in self.__names:
            current_alpha = self.run_data.get('alpha', name=name)
            current_target = self.run_data.get('target', name=name)
            self._logger.info("Plugin {}: alpha = {}, target = {}".format(
                name,
                current_alpha,
                current_target)
            )

    def run(self):

        phase = self.run_data.get('phase')

        self.__change_directory()

        if phase == 'training':
            self.__train()
            self.run_data.set(phase='convergence')

        elif phase == 'convergence':
            self.__converge()

            for i in range(len(self.__names)):
                data=self.data
                pair=self.pair
                A1=self.A1

                if data[i]<=25000 and data[i]>=15000:
                    self.A_parameter = 1
                    self.run_data.set(phase='production')
                else:
                    self.A_parameter = 0

                    if data[i]<1:
                        B = A1[i]*0.10
                        self.run_data.set(A =B, name=pair[i])
                        self.run_data.set(phase='training',start_time=0, iteration=self.run_data.get('iteration'))

                    elif data[i]<100 and data[i]>=1:
                        B = A1[i]*0.15
                        self.run_data.set(A = B, name=pair[i])
                        self.run_data.set(phase='training',start_time=0, iteration=self.run_data.get('iteration'))

                    elif data[i]<1000 and data[i]>=100:
                        B = A1[i]*0.20
                        self.run_data.set(A = B, name=pair[i])
                        self.run_data.set(phase='training',start_time=0, iteration=self.run_data.get('iteration'))

                    elif data[i]<10000 and data[i]>=1000:
                        B = A1[i]*0.25
                        self.run_data.set(A = B, name=pair[i])
                        self.run_data.set(phase='training',start_time=0, iteration=self.run_data.get('iteration'))

                    elif data[i]<12500 and data[i]>=10000:
                        B = A1[i]*0.75
                        self.run_data.set(A = B, name=pair[i])
                        self.run_data.set(phase='training',start_time=0, iteration=self.run_data.get('iteration'))

                    elif data[i]<15000 and data[i]>=12500:
                        B = A1[i]*0.90
                        self.run_data.set(A = B, name=pair[i])
                        self.run_data.set(phase='training',start_time=0, iteration=self.run_data.get('iteration'))

                    else: ##data>25000:##
                        B = A1[i]*1.3
                        self.run_data.set(A= B, name=pair[i])
                        self.run_data.set(phase='training',start_time=0, iteration=self.run_data.get('iteration'))


        else:
            self.__production()
            self.run_data.set(
                phase='training',
                start_time=0,
                iteration=(self.run_data.get('iteration') + 1))
        self.run_data.save_config(self.state_json)
