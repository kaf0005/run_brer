"""RunConfig class handles the actual workflow logic."""

from run_brer.run_data import RunData
from run_brer.pair_data import MultiPair
from run_brer.plugin_configs import TrainingPluginConfig, ConvergencePluginConfig, ProductionPluginConfig, PluginConfig
from run_brer.directory_helper import DirectoryHelper
from copy import deepcopy
import os
import shutil
import logging
import gmx
import json
import numpy as np
# import atexit

# This class encodes the numpy arrays of saved A values so that they may be added to the json dictionary
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class RunConfig:
    """Run configuration for single BRER ensemble member."""

    def __init__(self,
                 tpr,
                 ensemble_dir,
                 dict_json,
                 ensemble_num=1,
                 A_parameter=1,
                 pairs_json='pair_data.json',
                 sample_count=[],
                 retrain_count=0,
                 j=[]
                 ):
        """The run configuration specifies the files and directory structure
        used for the run. It determines whether the run is in the training,
        convergence, or production phase, then performs the run.

        Parameters
        ----------
        tpr : str
            path to tpr. Must be gmx 2017 compatible.
        ensemble_dir : str
            path to top directory which contains the full ensemble.
        ensemble_num : int, optional
            the ensemble member to run, by default 1
        pairs_json : str, optional
            path to file containing *ALL* the pair metadata.
            An example of what such a file should look like is provided in the d$
            by default 'pair_data.json'

        """

        self.tpr = tpr
        self.ens_dir = ensemble_dir
        self.dict_json=dict_json
        self.retrain_count=0
        self.j=[]
        self.sample_count=[]
        self.A_parameter=1

        # a list of identifiers of the residue-residue pairs that will be restrained
        self.__names = []

        # Load the pair data from a json. Use this to set up the run metadata
        self.pairs = MultiPair()
        self.pairs.read_from_json(pairs_json)
        # use the same identifiers for the pairs here as those provided in the pair metadata
        # file this prevents mixing up pair data amongst the different pairs (i.e.,
        # accidentally applying the restraints for pair 1 to pair 2.)
        self.__names = self.pairs.names

        self.run_data = RunData()
        self.run_data.set(ensemble_num=ensemble_num)
    
        self.state_json = '{}/mem_{}/state.json'.format(ensemble_dir, self.run_data.get('ensemble_num'))
        # If we're in the middle of a run, 
        #  the BRER checkpoint file and continue from
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
        formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        self._logger.addHandler(fh)
        self._logger.addHandler(ch)

        self._logger.info("Initialized the run configuration: {}".format(self.run_data.as_dictionary()))
        self._logger.info("Names of restraints: {}".format(self.__names))

        # Need to cleanly handle cancelled jobs: write out a checkpoint of the state if the
        # job is exited.
        # def cleanup():
        #     """"""
        #     self.run_data.save_config(self.state_json)
        #     self._logger.info("BRER received INT signal, stopping and saving data to {}".format(self.state_json))

        # atexit.register(cleanup)

    def build_plugins(self, plugin_config: PluginConfig):
        """Builds the plugin configuration. For each pair-wise restraint,
        populate the plugin with data: both the "general" data and the data
        unique to that restraint.

        Parameters
        ----------
        plugin_config : PluginConfig
            the particular plugin configuration (Training, Convergence, Production) for the run.
        """

        # One plugin per restraint.
        # TODO: what is the expected behavior when a list of plugins exists? Probably wipe them.
        self.__plugins = []
        general_params = self.run_data.general_params
        # For each pair-wise restraint, populate the plugin with data: both the "general" data and
        # the data unique to that restraint.
        for name in self.__names:
            pair_params = self.run_data.pair_params[name]
            new_restraint = deepcopy(plugin_config)
            new_restraint.scan_metadata(general_params)  # load general data into current restraint
            new_restraint.scan_metadata(pair_params)  # load pair-specific data into current restraint
            self.__plugins.append(new_restraint.build_plugin())

    def __change_directory(self):
        # change into the current working directory (ensemble_path/member_path/iteration/phase)
        dir_help = DirectoryHelper(top_dir=self.ens_dir, param_dict=self.run_data.general_params.get_as_dictionary())
        dir_help.build_working_dir()
        dir_help.change_dir('phase')

    def __move_cpt(self):
        current_iter = self.run_data.get('iteration')
        ens_num = self.run_data.get('ensemble_num')
        phase = self.run_data.get('phase')

        # If the cpt already exists, don't overwrite it
        if os.path.exists('{}/mem_{}/{}/{}/state.cpt'.format(self.ens_dir, ens_num, current_iter, phase)):
            self._logger.info("Phase is {} and state.cpt already exists: not moving any files".format(phase))

        else:
            member_dir = '{}/mem_{}'.format(self.ens_dir, ens_num)
            prev_iter = current_iter - 1

            if phase in ['training', 'convergence']:
                if prev_iter > -1:
                    # Get the production cpt from previous iteration
                    gmx_cpt = '{}/{}/production/state.cpt'.format(member_dir, prev_iter)
                    shutil.copy(gmx_cpt, '{}/state.cpt'.format(os.getcwd()))

                else:
                    pass  # Do nothing

            else:
                # Get the convergence cpt from current iteration
                gmx_cpt = '{}/{}/convergence/state.cpt'.format(member_dir, current_iter)
                shutil.copy(gmx_cpt, '{}/state.cpt'.format(os.getcwd()))

    def __moveDict(self):

        if self.dict_json==0:
            path='{}/mem_{}/dict.json'.format(self.ens_dir,self.run_data.get('ensemble_num'))
            if os.path.exists(path):
                self._logger.info("Dictionary already exists: not moving any files")

            else:
                #This is looking for the dict.json file from the previous ensemble 
                if os.path.exists('{}/mem_{}/dict.json'.format(self.ens_dir, (self.run_data.get('ensemble_num'))-1)):
                    json='{}/mem_{}/dict.json'.format(self.ens_dir,(self.run_data.get('ensemble_num')-1))
                    path='{}/mem_{}'.format(self.ens_dir, self.run_data.get('ensemble_num'))
                    shutil.copy(json, '{}'.format(path))
                    self._logger.info("Copying dictionary from previous ensemble number {}".format((self.run_data.get('ensemble_num'))-1))
                else:
                    self._logger.info("Dictionary not found; therefore, starting a new dictionary")
                    g = open('{}/mem_{}/dict.json'.format(self.ens_dir,self.run_data.get('ensemble_num')),'w+')
                    g.close()
        else:
            self._logger.info("Starting a new dictionary per user request")
            g = open('{}/mem_{}/dict.json'.format(self.ens_dir,self.run_data.get('ensemble_num')),'w+')
            g.close()     

                

    def __train(self):
        # do re-sampling
        targets = self.pairs.re_sample()
        self._logger.info('New targets: {}'.format(targets))
        for name in self.__names:
            self.run_data.set(name=name, target=targets[name])

        # save the new targets to the BRER checkpoint file.
        self.run_data.save_config(fnm=self.state_json)


        # backup existing checkpoint.
        # TODO: Don't backup the cpt, actually use it!!
        cpt = '{}/state.cpt'.format(os.getcwd())
        if os.path.exists(cpt):
            self._logger.warning(
                'There is a checkpoint file in your current working directory, but you are '
                'training. The cpt will be backed up and the run will start over with new targets'
            )
            shutil.move(cpt, '{}.bak'.format(cpt))

        # Checking the dictionary for an acceptable A value for the target, if found, it makes sure that that A value isn't in the rejectA dictionary
        path='{}/mem_{}/dict.json'.format(self.ens_dir,self.run_data.get('ensemble_num'))
        for name in self.__names:
            with open(path,'r+') as f:
                if os.path.getsize(path)==0:
                    pass #nothing in dictionary
                else:
                    dict=json.load(f)
                    current_target=self.run_data.get('target',name=name)
                    current_target='{:2f}'.format(current_target)

                    if current_target in dict['{}'.format(name)]['acceptA'].keys():
                        possible_A=dict.get('{}'.format(name), {}).get('acceptA',{}).get('{}'.format(current_target))
                        possible_A=np.array(possible_A)
                        A = np.sort(possible_A, axis=None)
                        A=np.median(A)

                        if current_target in dict['{}'.format(name)]['rejectA'].keys():
                            possible_A=dict.get('{}'.format(name), {}).get('rejectA',{}).get('{}'.format(current_target))
                            A1=np.array(possible_A)
                            if A in A1:
                                A=1.1*A                        
                                self.run_data.set(A=A,name=name)
                            else:
                                self.run_data.set(A=A, name=name)
                    else:
                        pass #use the default A-value
                
 
        # Set up a dictionary to go from plugin name -> restraint name
        sites_to_name = {}

        # Build the gmxapi session.
        md = gmx.workflow.from_tpr(self.tpr, append_output=False)
        self.build_plugins(TrainingPluginConfig())
        for plugin in self.__plugins:
            plugin_name = plugin.name
            for name in self.__names:
                run_data_sites = "{}".format(self.run_data.get('sites', name=name))
                if run_data_sites == plugin_name:
                    sites_to_name[plugin_name] = name
            md.add_dependency(plugin)
        context = gmx.context.ParallelArrayContext(md, workdir_list=[os.getcwd()])

        # Run it.
        with context as session:
            session.run()

        # In the future runs (convergence, production) we need the ABSOLUTE VALUE of alpha.
        self._logger.info("=====TRAINING INFO======\n")

        for i in range(len(self.__names)):
            current_name = sites_to_name[context.potentials[i].name]
            current_alpha = context.potentials[i].alpha
            current_target = context.potentials[i].target

            self.run_data.set(name=current_name, alpha=current_alpha)
            self.run_data.set(name=current_name, target=current_target)
            self._logger.info("Plugin {}: alpha = {}, target = {}".format(current_name, current_alpha, current_target))

    def __retrain(self):
        self.run_data.from_dictionary(json.load(open(self.state_json)))
        corr_target=[]
        count=0
        names=[]
        if self.retrain_count == 6:
            self.retrain_count =1

        for name in self.__names:
            # Setting to original targets
            corr_target.append(self.run_data.get('target', name=name))
            self.run_data.set(name=name, target=corr_target[count])
            names.append(name)
            names.append(corr_target[count])
            count=count+1
        self._logger.info('Original targets: {}'.format(names))
        self.run_data.save_config(fnm=self.state_json)

        # This reads through the memory from dict.json and determines if A is suitable for the retrain
        path='{}/mem_{}/dict.json'.format(self.ens_dir,self.run_data.get('ensemble_num'))
        for name in self.__names:
            with open(path,'r+') as f:
                if os.path.getsize(path)==0:
                    pass #nothing in dictionary
                else:
                    dict=json.load(f)
                    current_target=self.run_data.get('target',name=name)
                    current_target='{:2f}'.format(current_target)

                    if current_target in dict['{}'.format(name)]['acceptA'].keys():
                        possible_A=dict.get('{}'.format(name), {}).get('acceptA',{}).get('{}'.format(current_target))
                        possible_A=np.array(possible_A)
                        A = np.sort(possible_A, axis=None)
                        A=np.median(A)

                        if current_target in dict['{}'.format(name)]['rejectA'].keys():
                            possible_A=dict.get('{}'.format(name), {}).get('rejectA',{}).get('{}'.format(current_target))
                            A1=np.array(possible_A)
                            
                            if A in A1:
                                A=1.1*A                        
                                self.run_data.set(A=A,name=name)
                            else:
                                self.run_data.set(A=A, name=name)
                    else:
                        pass #use the default A-value
            
        # backup existing checkpoint.
        # TODO: Don't backup the cpt, actually use it!!
        cpt = '{}/state.cpt'.format(os.getcwd())
        if os.path.exists(cpt):
            self._logger.warning(
                'There is a checkpoint file in your current working directory; however you are '
                'training with original targets. The cpt will be backed up and the run will start over with original targets.'
            )
            shutil.move(cpt, '{}.bak'.format(cpt))

        # Set up a dictionary to go from plugin name -> restraint name
        sites_to_name = {}

        # Build the gmxapi session.
        md = gmx.workflow.from_tpr(self.tpr, append_output=False)
        self.build_plugins(TrainingPluginConfig())
        for plugin in self.__plugins:
            plugin_name = plugin.name
            for name in self.__names:
                run_data_sites = "{}".format(self.run_data.get('sites', name=name))
                if run_data_sites == plugin_name:
                    sites_to_name[plugin_name] = name
            md.add_dependency(plugin)
        context = gmx.context.ParallelArrayContext(md, workdir_list=[os.getcwd()])

        # Run it.
        with context as session:
            session.run()

        # In the future runs (convergence, production) we need the ABSOLUTE VALUE of alpha.
        self._logger.info("=====TRAINING INFO======\n")

        for i in range(len(self.__names)):
            current_name = sites_to_name[context.potentials[i].name]
            current_alpha = context.potentials[i].alpha
            current_target = context.potentials[i].target

            self.run_data.set(name=current_name, alpha=current_alpha)
            self.run_data.set(name=current_name, target=current_target)
            self._logger.info("Plugin {}: alpha = {}, target = {}".format(current_name, current_alpha, current_target))

    def __datDict(self):
        # Reads in dictionary, if there are no values it sets up the nested dictionary
        path = '{}/mem_{}/dict.json'.format(self.ens_dir, self.run_data.get('ensemble_num'))
        with open(path, 'r+') as g: 
            if os.path.getsize(path)>0: 
                dict=json.load(g)
                j=json.dumps(dict)
                self.j=j
            else:
                dict={}
                for name in self.__names:
                    dict['{}'.format(name)]={}
                    dict['{}'.format(name)]['acceptA']={}
                    dict['{}'.format(name)]['rejectA']={}
                j=json.dumps(dict)
                self.j=j

        self.A_parameter=1
        self.sample_count=0
        for name in self.__names: 
            dict=json.loads(self.j)
            path='{}/mem_{}/{}/training/{}.log'.format(self.ens_dir,self.run_data.get('ensemble_num'),self.run_data.get('iteration'), name)   
            if os.path.exists(path):
                with open('{}/mem_{}/{}/training/{}.log'.format(self.ens_dir,self.run_data.get('ensemble_num'),self.run_data.get('iteration'), name),"r") as openfile:
                    f=openfile.readlines()[-1]
                    f=f.replace('\t',',')
                    f=np.matrix(f)
                    sample_count=f[0,2]
                    corr_target= f[0,3]
                    corr_target='{:2f}'.format(corr_target)
                    corr_A  = self.run_data.get('A',name=name)  
                    # Resets the A value if the training did not converge within 20ns, these values are saved in the dictionary 
                    if sample_count >400:
                        self.sample_count=sample_count
                        # Reassigning the A-value
                        A=self.run_data.get('A', name=name)
                        if self.retrain_count==5:
                            A=2*A
                        else:
                            A=1.1*A
                            self.run_data.set(A=A, name=name)
                        if corr_target in dict['{}'.format(name)]['rejectA'].keys():
                            A=dict.get('{}'.format(name),{}).get('rejectA',{}).get('{}'.format(corr_target))
                            A=np.array(A)
                            A=np.append(A,corr_A)
                            dict['{}'.format(name)]['rejectA']['{}'.format(corr_target)]=A
                            j=json.dumps(dict,cls=NumpyEncoder)
                        else:
                            dict['{}'.format(name)]['rejectA']['{}'.format(corr_target)]=corr_A
                            j=json.dumps(dict)

                    else:
                        A=self.run_data.get('A', name=name)
                        if corr_target in dict['{}'.format(name)]['acceptA'].keys():
                            A=dict.get('{}'.format(name),{}).get('acceptA',{}).get('{}'.format(corr_target))
                            A=np.array(A)
                            A=np.append(A,corr_A)
                            dict['{}'.format(name)]['acceptA']['{}'.format(corr_target)]=A
                            j=json.dumps(dict,cls=NumpyEncoder)
                        else:
                            dict['{}'.format(name)]['acceptA']['{}'.format(corr_target)]=corr_A
                            j=json.dumps(dict)
                self.j=j

        with open('{}/mem_{}/dict.json'.format(self.ens_dir,self.run_data.get('ensemble_num')), 'w+') as g:
            g.write(self.j)


    def __converge(self):
        self.__move_cpt()

        md = gmx.workflow.from_tpr(self.tpr, append_output=False)
        self.build_plugins(ConvergencePluginConfig())
        for plugin in self.__plugins:
            md.add_dependency(plugin)
        context = gmx.context.ParallelArrayContext(md, workdir_list=[os.getcwd()])
        with context as session:
            session.run()

	# Get the absolute time (in ps) at which the convergence run finished.
        # This value will be needed if a production run needs to be restarted.

        self.run_data.set(start_time=context.potentials[0].time)

        self._logger.info("=====CONVERGENCE INFO======\n")
        for name in self.__names:
            current_alpha = self.run_data.get('alpha', name=name)
            current_target = self.run_data.get('target', name=name)
            self._logger.info("Plugin {}: alpha = {}, target = {}".format(name, current_alpha, current_target))

    def __production(self):

        # Get the checkpoint file from the convergence phase
        self.__move_cpt()

        # Calculate the time (in ps) at which the BRER iteration should finish.
        # This should be: the end time of the convergence run + the amount of time for
        # production simulation (specified by the user).
        end_time = self.run_data.get('production_time') + self.run_data.get('start_time')

        md = gmx.workflow.from_tpr(self.tpr, end_time=end_time, append_output=False)

        self.build_plugins(ProductionPluginConfig())
        for plugin in self.__plugins:
            md.add_dependency(plugin)
        context = gmx.context.ParallelArrayContext(md, workdir_list=[os.getcwd()])
        with context as session:
            session.run()

        self._logger.info("=====PRODUCTION INFO======\n")
        for name in self.__names:
            current_alpha = self.run_data.get('alpha', name=name)
            current_target = self.run_data.get('target', name=name)
            self._logger.info("Plugin {}: alpha = {}, target = {}".format(name, current_alpha, current_target))

    def run(self):
        """Perform the MD simulations.
        """
        phase = self.run_data.get('phase')
        self.__change_directory()
     
        if phase == 'training':
            self.__moveDict()
            self.__train()
            self.__datDict()
            if self.sample_count>400:
                while self.sample_count>400:
                    self.retrain_count=self.retrain_count+1
                    self.__retrain()
                    self.__datDict()
                self.run_data.set(phase='convergence') 
            else:
                self.run_data.set(phase='convergence')
            
        elif phase == 'convergence':
            self.__converge()
            self.run_data.set(phase='production')
        
        else:
            self.__production()
            self.run_data.set(phase='training', start_time=0, iteration=(self.run_data.get('iteration') + 1))
        self.run_data.save_config(self.state_json)


