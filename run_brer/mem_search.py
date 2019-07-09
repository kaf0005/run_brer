
#!/usr/bin/env python3
from run_brer.run_data import RunData
from run_brer.pair_data import MultiPair
import os
import json 
import numpy as np 
import sys


class Analysis: 

    def __init__(self,
                ensemble_dir,
                analysis_dir,
                n,
                m,
                pairs_json='pair_data.json',
                data=[],
                W_matrix=[],
                distance_values=[]
                ):

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
        self.distance_values=distance_values

    def __gromacs(self):
        os.system('echo arg1 arg2 | gmx trjconv -f {traj_comp.part000*} -o {combine_traj.xtc}'.format(other, arguments))

    def __awk(self):
        with open("distance.xvg","r+") as f:
            data=f.read().splitlines(True)
        with open("dist_trial.log","w") as fout:
            fout.writelines(data[17:])
        with open("dist_trial.log","r+") as g:
            data=g.read()
            print(data)
            data=data.replace('\n',';')
            data=data.replace(' ',',')
            data=data.strip(',;')
        distance_values = np.matrix(data)
        self.distance_values=distance_values
    

    def __logData(self):
        #use this when in training directory
        save_data=np.zeros((3,3))
        count=0
        for name in self.__names:
            with open("{}.log".format(name),"r+") as f:
                data=f.readlines()[-1]
                data=data.replace(' ',',')
                data=np.matrix(data)
                save_data[count,2]=data[0,5] #alpha
                save_data[count,0]=data[0,3] #target
                save_data[count,1]=data[0,2] #R
                count=count+1
        self.data=save_data
    
    def __workCalc(self):
        distance_values=self.distance_values
        n=distance_values.shape[0]
        data=self.data
        n=n-1
        W_DEER1=0
        W_DEER2=0
        W_DEER3=0
        for i in range (0,n):
            Sum1=data[0,2]*(distance_values[i+1,1]-distance_values[i,1])
            W_DEER1 =W_DEER1+Sum1
            Sum2=data[1,2]*(distance_values[i+1,2]-distance_values[i,1])
            W_DEER2=W_DEER2+Sum2
            Sum3=data[2,2]*(distance_values[i+1,3]-distance_values[i,1])
            W_DEER3=W_DEER3+Sum3

            W_DEER1=(W_DEER1)*1e-12
            W_DEER2=(W_DEER2)*1e-12
            W_DEER3=(W_DEER3)*1e-12
            
            W_matrix=[W_DEER1,W_DEER2,W_DEER3]
            W_matrix=np.matrix(W_matrix)
            self.W_matrix=W_matrix
    

    def __binSet(self):
        #to be ran after all of the mem_directories have gone through 
        data=self.data
        W_matrix=self.W_matrix
        name_dict=str(data[0,1])+"_"+str(data[1,1])+"_"+str(data[2,1])+".dat"
        if name_dict in self.analysis_dir:
            with open(name_dict,"a+") as g:                
                g.write('%f' % W_matrix)
            g.close()
        else: 
            with open(name_dict,"w+") as g:
                g.write("Work for each DEER distribution in kJ\n")
                g.write('%f' % W_matrix)
            g.close()

        
    def run(self):
        n=self.n
        m=self.m
        for i in range(n):
            for j in range(m):
                self.n=i
                self.m=j
                path="../mem_{}/{}".format(self.n,self.m)
                if os.path.exists(path):
                    os.chdir(path)
                    self.__gromacs
                    self.__awk
                    self.__logData
                    self.__workCalc
                    self.__binSet

                else:
                    pass #do nothing because mem directory or iteration does not exist