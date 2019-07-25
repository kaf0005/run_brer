
import gmxapi as gmx
import glob
import os 

path="/home/kaf0005/UVA_reu/test-brer/mem_1/0/convergence" 
os.chdir(path) #convergence directory 
list=os.listdir()
traj=glob.glob("{}/*.xtc".format(path))
# for names in list:
#     if names.endswith(".xtc"):
#         traj.append(names)

trjcat=gmx.commandline_operation(
                'gmx',
                'trjcat',
                input_files={'-f': '/path/to/this/traj_comp.part0001.xtc'},
                output_files={'-o': '{}/trjcat.xtc'.format(os.getcwd())}
                )
print(traj)
print(glob.glob("{}/*.xtc".format(path)))