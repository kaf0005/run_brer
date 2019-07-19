
#!/usr/bin/env python3
from run_brer.run_data import RunData
from run_brer.pair_data import MultiPair
import os
import json 
import numpy as np 
import sys
import datetime
import codecs


class Analysis: 

    def __init__(self,
                tpr,
                index_file,
                select,
                ensemble_dir,
                analysis_dir,
                n,
                m,
                pairs_json='pair_data.json',
                data=[],
                W_matrix=[],
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
        self.__names=[]
        self.pairs = MultiPair()
        self.pairs.read_from_json(pairs_json)
        self.__names = self.pairs.names
        self.state_json = '{}/mem_{}/state.json'.format(self.ensemble_dir,self.n)
        self.data=[]
        self.W_matrix=[]
        self.distance_values=[]
        self.end_signal=0

    def __gromacs(self):
        import gromacs
        path="{}/mem_{}/{}/convergence".format(self.ensemble_dir, self.n,self.m) 
        os.chdir(path) #convergence directory 
        list=os.listdir()
        traj=[]
        for names in list:
            if names.endswith(".xtc"):
                traj.append(names)
        if not traj:
            self.end_signal=1
        else:
            combined_traj="{}/combined_traj.xtc".format(self.analysis_dir)
            gromacs.trjcat(f=traj, o = combined_traj)
            path="{}".format(self.analysis_dir)
            os.chdir(path) #analysis directory
            path=self.index_file
            if os.path.exists(path):
                gromacs.distance(f="combined_traj.xtc", s=self.tpr, n=self.index_file, oall="distance.xvg", select=self.select)
            else: 
                gromacs.distance(f="combined_traj.xtc", s=self.tpr, oall="distance.xvg")
                

    def __awk(self):
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
        os.remove("combined_traj.xtc")
    

    def __logData(self):
        save_data=[]
        count=0
        path="{}/mem_{}/{}/training".format(self.ensemble_dir, self.n, self.m) #training directory
        os.chdir(path)
        for name in self.__names:
            with open("{}.log".format(name),"r+") as f:
                data=f.readlines()[-1]
                data=data.replace(' ',',')
                data=np.matrix(data)
                save_data.append(data[0,2]) 
                save_data.append(data[0,1]) 
                save_data.append(data[0,4]) 
                count=count+1
        save_data=np.matrix(save_data)    
        self.data=save_data
           
    
    def __workCalc(self):
        distance_values=self.distance_values
        distance_values=np.matrix(distance_values)
        n=distance_values.shape[0]
        data=self.data
        n=n-1
        W_DEER1=0
        W_DEER2=0
        W_DEER3=0
        for i in range (0,n):
            Sum1=data[0,2]*(distance_values[i+1,1]-distance_values[i,1])
            W_DEER1 =W_DEER1+Sum1
            Sum2=data[0,5]*(distance_values[i+1,2]-distance_values[i,2])
            W_DEER2=W_DEER2+Sum2
            Sum3=data[0,8]*(distance_values[i+1,3]-distance_values[i,3])
            W_DEER3=W_DEER3+Sum3

            W_DEER1=(W_DEER1)*1e-12
            W_DEER2=(W_DEER2)*1e-12
            W_DEER3=(W_DEER3)*1e-12
            
            W_matrix=[W_DEER1,W_DEER2,W_DEER3]
            W_matrix=np.matrix(W_matrix)
            self.W_matrix=W_matrix

    def __datDict(self):         
        # Reads in dictionary, if there are no values it sets up the nested dictionary
        data=self.data
        targetSet='{:2f}_{:2f}_{:2f}'.format(data[0,0],data[0,3],data[0,6])
        workCalc=np.sum(self.W_matrix)
        path = '{}/targetSet.json'.format(self.analysis_dir)
        if os.path.exists(path):
            if os.path.getsize(path)>0:
                with open('{}/targetSet.json'.format(self.analysis_dir), "r+") as f:
                    dict=json.load(f)
                if '{}'.format(targetSet) in dict:
                    values=dict.get('{}'.format(targetSet))
                    values=np.array(values)
                    values=np.append(values,workCalc)
                    dict['{}'.format(targetSet)]=values
                else:
                    dict['{}'.format(targetSet)]=workCalc
            else:
                dict={}
                dict['{}'.format(targetSet)]=workCalc
        with open('{}/targetSet.json'.format(self.analysis_dir), "w+") as f:
            j=json.dumps(dict)
            f.write(j)      

    def __analysisLog(self):
        path=('{}/mem_{}/{}'.format(self.ensemble_dir,self.n,self.m))
        os.chdir(path)
        now=datetime.datetime.now()
        with open("analysis.log","w+") as f:
            f.write("Analysis was completed on:\t")
            f.write(now.strftime("%Y-%m-%d"))
            f.write("\tat:\t")
            f.write(now.strftime("%H:%M"))

    def run(self):
        n=self.n
        m=self.m
        for i in n:
            for j in m:
                self.n=i
                self.m=j
                path="{}/mem_{}/{}/convergence/md.part0001.log".format(self.ensemble_dir,self.n,self.m)
                if os.path.exists(path):
                    exists=os.path.isfile('{}/mem_{}/{}/analysis.log'.format(self.ensemble_dir,self.n,self.m))
                    if exists:
                        print("You have already done analysis on this iteration:")
                        print('{}/mem_{}/{}'.format(self.ensemble_dir,self.n,self.m))
                        print("If you wish to do another analysis run on this mem_directory and iteration,")
                        print("please delete the analysis.log file within the iteration directory.")
                        print("\n")
                        break 
                    else:   
                        self.__gromacs()
                        if self.end_signal==1:
                            print("There were no xtc files found in the following mem_directory:")
                            cwd=os.getcwd()
                            print(cwd)
                            print("You may need to do another ./run.py to continue that run.")
                            print("\n")
                            break

                        self.__awk()
                        self.__logData()
                        self.__workCalc()
                        self.__datDict()
                        self.__analysisLog()
                else:
                    print("There are no md log files in the convergence folder for the current iteration:")
                    print(path)
                    print("You may need to do another ./run.py to continue that run.")
                    print("\n")
                    pass #do nothing because mem directory or iteration does not exist
