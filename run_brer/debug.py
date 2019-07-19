
#!/usr/bin/python
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

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class Run:
    def __init__(self,
                 tpr,
                 ensemble_dir,
                 dict_json,
                 ensemble_num=1,
                 pairs_json='pair_data.json',
                 A_parameter=[],
                 retrain_count=0,
                 targetSet=[],
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
        self.A_parameter = []
        self.retrain_count=0
        self.targetSet=[]
        self.j=[]


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
    def __retrain(self):
        self.run_data.from_dictionary(json.load(open(self.state_json)))
        corr_target=[]
        count=0
        names=[]
        if self.retrain_count == 6:
            self.retrain_count =0

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
        for name in self.__names:
            with open('{}/mem_{}/dict.json'.format(self.ens_dir,self.run_data.get('ensemble_num')),'r+') as f:
                try:
                    dict=json.loads(self.j)
                    current_target=self.run_data.get('target',name=name)
                    current_target_dict='{:2f}'.format(current_target)
                    possible_target=dict['{}'.format(name)]['acceptA']
                    possible_target=list(possible_target.keys())
                    possible_target=''.join(possible_target)
                    possible_target=float(possible_target)
                    possible_target=np.matrix(possible_target)

                    if current_target in possible_target:
                        target_index =   np.where(possible_target==current_target)[0]
                        target_index=int(target_index)
                        possible_A=dict.get('{}'.format(name), {}).get('acceptA',{}).get('{}'.format(current_target_dict))
                        possible_A=np.array(possible_A)
                        A = possible_A[target_index]
                        A=np.array(A)
                        A=np.sort(A,axis=None)
                        A=np.median(A)
                        
                        try:
                            possible_target=dict['{}'.format(name)]['rejectA']
                            possible_target=list(possible_target.keys())
                            possible_target=''.join(possible_target)
                            possible_target=float(possible_target)
                            possible_target=np.matrix(possible_target)
                            print('\n')
                            if current_target in possible_target:
                                target_index =   np.where(possible_target==current_target)
                                target_index=int(target_index)
                                possible_A=dict.get('{}'.format(name), {}).get('rejectA',{}).get('{}'.format(current_target_dict))
                                possible_A=np.array(possible_A)
                                A1 = possible_A[target_index]
                                A1=np.array(A1)
                                i=len(A)
                                i=i-1
                                j=len(A1)
                                j=j-1
                                break_loop=0
                                for i in range(0,i):
                                    if break_loop==1:
                                        break
                                    else:
                                        continue
                                    for j in range(0,j):
                                        if A[i]==A1[j]:
                                            A=1.1*A
                                            break_loop=1                        
                                self.run_data.set(A=A,name=name)
                            else:
                                self.run_data.set(A=A, name=name)
                        except KeyError:
                            pass #retrain KeyError: rejectA for the current target does not exist yet
                    else:
                        pass #use the default A-value
                             
                except KeyError:
                    pass #retrain KeyError: acceptA for the current target does not exist yet
                except ValueError: 
                    pass #retrain ValueError: dictionary values do not exist yet
                           

    def datDict(self):
            # Reads in dictionary, if there are no values it sets up the nested dictionary
        path = '{}/mem_{}/dict.json'.format(self.ens_dir, self.run_data.get('ensemble_num'))
        with open('{}/mem_{}/dict.json'.format(self.ens_dir,self.run_data.get('ensemble_num')), 'r+') as g: 
            try:
                dict_original=json.load(g)
                print(dict_original)
                dict=dict_original
                j=json.dumps(dict)
                self.j=j
                self.dict_json=1
                merge=1

            except ValueError:
                dict={}
                for name in self.__names:
                    dict['{}'.format(name)]={}
                    dict['{}'.format(name)]['acceptA']={}
                    dict['{}'.format(name)]['rejectA']={}
                    #dict.update(d)
                    #d={'{}'.format(name):{'rejectA':[]}
                    j=json.dumps(dict)
                print(j)
                j=json.dumps(dict)
                self.j=j
                self.dict_json=0
                merge=0

        print(merge)
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
                    sample_count =1
                    if sample_count >0:
                        self.sample_count=sample_count
                        # Reassigning the A-value
                        A=self.run_data.get('A', name=name)
                        if self.retrain_count==5:
                            A=2*A
                        else:
                            A=1.1*A
                            self.run_data.set(A=A, name=name)
                        try:
                            values=dict.get('{}'.format(name), {}).get('rejectA',{}).get('{}'.format(corr_target))
                            if values == None:
                                dict['{}'.format(name)]['rejectA']['{}'.format(corr_target)]=corr_A
                                j=json.dumps(dict) 
                            else: 
                                values=np.array(values)
                                values=np.append(values, corr_A)
                                dict['{}'.format(name)]['rejectA']['{}'.format(corr_target)]=values
                                j=json.dumps(dict,cls=NumpyEncoder)
                
                        except KeyError:
                            dict['{}'.format(name)]['rejectA']['{}'.format(corr_target)]=corr_A
                            j=json.dumps(dict)

                    else:
                        A=self.run_data.get('A', name=name)
                        try:
                            values=dict.get('{}'.format(name), {}).get('acceptA',{}).get('{}'.format(corr_target))
                            if values == None:
                                d={'{}'.format(name):{'acceptA':{'{}'.format(corr_target):[corr_A]}}}
                                dict.update(d)
                                j=json.dumps(dict)
                            else:
                                values=np.array(values)
                                values=np.append(values,corr_A)
                                dict['{}'.format(name)]['acceptA']['{}'.format(corr_target)]=values
                                j=json.dumps(dict,cls=NumpyEncoder)

                        except KeyError:
                            dict=['{}'.format(name)]['acceptA']['{}'.format(corr_target)]=corr_A
                            j=json.dumps(dict)
                self.j=j
                print(j)

        with open('{}/mem_{}/dict.json'.format(self.ens_dir,self.run_data.get('ensemble_num')), 'w+') as g:
            if merge ==0:
                g.write(self.j)
            else:
                dict=json.loads(self.j) # fix this         
                #dict={**dict_original,**self.j} #merging the two different dictionaries
                j=json.dumps(dict)
                g.write(self.j)
            self._logger.info('{}'.format(dict))
     #       self._logger.info('{}'.format(dict))
      #self.__retrain()

