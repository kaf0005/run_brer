
#!/usr/bin/env python3
from run_brer.run_data import RunData
from run_brer.pair_data import MultiPair
import os
import json 
import numpy as np 
import sys
import datetime
import gmxapi as gmx 
import glob

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class Analysis: 

    def __init__(self,tpr,index_file,
                select,
                ensemble_dir,
                analysis_dir,
                n,
                m,
                pairs_json='pair_data.json',
                data=[],
                boltzman=[],
                distance_values=[],
                end_signal=[]
                ):
        self.tpr=tpr
        self.index_file=index_file
        self.select=select
        self.ensemble_dir=ensemble_dir
        self.analysis_dir=analysis_dir
        self.n=n
        self.m=m

        self.pairs = MultiPair()
        self.pairs.read_from_json(pairs_json)
        self.__names = self.pairs.names

        
        self.data=[]
        self.distance_values=[]
        self.end_signal=0

        self.run_data = RunData()

    def __gromacs(self,i,j):
        path="{}/mem_{}/{}/convergence".format(self.ensemble_dir, i,j) 
        os.chdir(path) #convergence directory 
        list=os.listdir()
        for names in list:
            if names.endswith(".xtc"):
                traj=glob.glob("{}/*.xtc".format(path))
        if not traj:
            self.end_signal=1
        else:
            trjcat=gmx.commandline_operation(
                'gmx',
                'trjcat',
                input_files={'-f': traj},
                output_files={'-o': '{}/trjcat.xtc'.format(self.analysis_dir)}
                )
            trjcat.run()
        
            path='{}'.format(self.analysis_dir)
            os.chdir(path) #analysis directory
            path=self.index_file
            if os.path.exists(path):
                distance=gmx.commandline_operation(
                    'gmx',
                    'distance',
                    input_files={
                            '-f': 'trjcat.xtc',
                            '-s': '{}'.format(self.tpr), 
                            '-n': '{}'.format(self.index_file),
                            '-select': '{}'.format(self.select)},
                    output_files={'-oall':'distance.xvg'} 
                    )
                distance.run()                
            else:
                distance=gmx.commandline_operation(
                    'gmx',
                    'distance',
                    input_files={
                            '-f': 'combined_traj.xtc',
                            '-s': '{}'.format(self.tpr)},
                    output_files={'-oall':'distance.xvg'} 
                    )   
                distance.run()
                
    def __awk(self):
        # this removes the unnecessary gromacs header in the .xvg file and writes just the data portion to dist.log
        path="{}".format(self.analysis_dir)
        os.chdir(path)
        with open("distance.xvg","r+") as f:
            data=f.read().splitlines(True)
        with open("dist.log","w") as fout:
            fout.writelines(data[17:])
        with open("dist.log","r+") as g:
            data=g.read()
            data=data.replace('\n',';')
            data=data.replace(' ',',')
            data=data.strip(',;')
        distance_values = np.matrix(data)
        self.distance_values=distance_values
        os.remove("distance.xvg")
        os.remove("trjcat.xtc")
    

    def __logData(self,i,j):
        #logData saves the target, R and alpha from the log files in training directories
        save_data=[]
        path="{}/mem_{}/{}/training".format(self.ensemble_dir, i, j) #training directory
        os.chdir(path)
        for name in self.__names:
            with open("{}.log".format(name),"r+") as f:
                data=f.readlines()[-1]
                data=data.replace(' ',',')
                data=np.matrix(data)
                save_data.append(data[0,3]) #target (nm)
                save_data.append(data[0,1]) #R (nm)
                save_data.append(data[0,5]) #alpha (kJ/mol)
        save_data=np.matrix(save_data)    
        self.data=save_data
           
    
    def __workCalc(self):
        #This calculates the work in kJ/mol
        work={}
        RT=2.479 #kJ/mol;
        distance_values=self.distance_values
        #distance_values=np.matrix(distance_values)
        n=distance_values.shape[0]
        n=n-1
        data=self.data
        m=0
        i=2
        j=0
        for name in self.__names:
            m=m+1
            alpha=data[0,i]
            target=data[0,j]
            force_constant=alpha/target #(kJ/mol)/nm
            i=i+3
            j=j+3
            W_DEER=0
            for k in range (0,n):
                delta_R=np.sum(np.abs(distance_values[k+1,m]-distance_values[k,m]))
                W_DEER =W_DEER+Sum
            W_DEER=force_constant*W_DEER
            work[name] =W_DEER

        #Calculating boltzmann
        z=np.sum(list(work.values()))
        self.boltzman=np.exp((-z)/RT)

        #Printing the work values on the command line
        data=self.data
        targetSet='{:2f}_{:2f}_{:2f}'.format(data[0,0],data[0,3],data[0,6])
        print('Work in kJ/mol for targetset: {}'.format(targetSet))
        print(z)

    def __datDict(self):         
        # Reads in dictionary, if there are no values it sets up the nested dictionary
        data=self.data
        targetSet='{:2f}_{:2f}_{:2f}'.format(data[0,0],data[0,3],data[0,6])
        path = '{}/targetSet.json'.format(self.analysis_dir)
        if os.path.exists(path):
            if os.path.getsize(path)>0:
                with open('{}/targetSet.json'.format(self.analysis_dir), "r+") as f:
                    dict=json.load(f)
                    if targetSet in dict.keys():
                        values=dict.get('{}'.format(targetSet))
                        values=np.array(values)
                        values=np.append(values,self.boltzman)
                        dict['{}'.format(targetSet)]=values
                        j=json.dumps(dict,cls=NumpyEncoder)
                    else:
                        dict['{}'.format(targetSet)]=self.boltzman
                        j=json.dumps(dict)
            else:
                dict={}
                dict['{}'.format(targetSet)]=self.boltzman
                j=json.dumps(dict)
        else:
            dict={}
            dict['{}'.format(targetSet)]=self.boltzman
            j=json.dumps(dict)
                
        with open('{}/targetSet.json'.format(self.analysis_dir), "w+") as f:
            f.write(j)
                

    def __analysisLog(self,i,j):
        path=('{}/mem_{}/{}'.format(self.ensemble_dir,i,j))
        os.chdir(path)
        now=datetime.datetime.now()
        with open("analysis.log","w+") as f:
            f.write("Analysis was completed on:\t")
            f.write(now.strftime("%Y-%m-%d"))
            f.write("\tat:\t")
            f.write(now.strftime("%H:%M"))

    def run(self):
        for i in self.n:
            for j in self.m:
                self.state_json = '{}/mem_{}/state.json'.format(self.ensemble_dir,i)
                path="{}/mem_{}/{}/convergence/md.part0001.log".format(self.ensemble_dir,i,j)
                if os.path.exists(path):
                    exists=os.path.isfile('{}/mem_{}/{}/analysis.log'.format(self.ensemble_dir,i,j))
                    if exists:
                        print("You have already done analysis on this iteration:")
                        print('{}/mem_{}/{}'.format(self.ensemble_dir,i,j))
                        print("If you wish to do another analysis run on this mem_directory and iteration,")
                        print("please delete the analysis.log file within the iteration directory.")
                        print("\n")
                        break 
                    else:   
                        self.__gromacs(i,j)
                        if self.end_signal==1:
                            print("There were no xtc files found in the following mem_directory:")
                            cwd=os.getcwd()
                            print(cwd)
                            print("You may need to do another ./run.py to continue that run.")
                            print("\n")
                            break

                        self.__awk()
                        self.__logData(i,j)
                        self.__workCalc()
                        self.__datDict()
                        self.__analysisLog(i,j)
                else:
                    print("There are no md log files in the convergence folder for the current iteration:")
                    print(path)
                    print("You may need to do another ./run.py to continue that run.")
                    print("\n")
