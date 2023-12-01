# PNPMT
Deep Plug-and-Play Prior for Multitasking Channel Reconstruction in Massive MIMO Systems

   CONTENTS：
（A）Files introduction
（B）Envoriment configuration
（C）Parameters setting

---------------------------------------------------------------------------------------------------
（A） Files introduction
This package PNPMT includes three different demos of downlink channel reconstruction tasks mentioned in the paper, which are channel estimation, antenna extrapolation, and CSI feedback respectively. 

1) The programm file named "pppce_demo.py" gives a demonstration of one sample in channal estimation task. 
2) The programm file named "pppe_demo.py" gives a demonstration of one sample in antenna extrapolation task.
3) The programm file named "pppcf_demo.py" gives a demonstration of 100 samples (equals to the number of samples in one block) in CSI feedback task.
4) The folder "data_npz" contains four datasets which are "pppce_data.npz", "pppae_data.npz", "pppcf_data.npz" used in three demos respectively, and "random_matrix.npz" only for the CSI feedback task. 
5) The folder "model_zoo" contains a single well-trained deep learning model "ffdnet.h5" which is employed to solve the subproblem (17b)  for three tasks. It is a DL-based denoiser and is going to be loaded in three demos.
6) The folder "utils" contains several functional files which are "util_metric.py", "util_module.py", "util_norm.py" and "util_pnp.py".

------------------------------------------------------------------------------------------------------
（B）Envoriment configuration
The environment to run these codes is suggested as "python 3.7.11 + tensorflow 2.5.2". Other environments are not precluded but be sure to configure a correct and workable enviroment.

-------------------------------------------------------------------------------------------------------
（C）Parameters setting
To demonstrate the effectiveness of the multi-task deep PnP prior method, three demos with a set of specific parameters are provided to run. The parameters for each demo are set as follows:
1)  pppce_demo.py

In the task of channel estimation, the number of pilots and the mode of pilots are adjustable.

Number of pilot: 256

mode: 1 

model name: fddnet 

Number of iterations: 10  

2) pppae_demo.py

In the task of antenna extrapolation, the number of antennas and the mode of antennas are adjustable.

Number of antenna: 16

mode: 1

model name: fddnet

Number of iterations: 10

3) pppcf_demo.py

In the task of CSI feedback, the compression ratio and quantification bits are adjustable.

encode_dim: 256

quan_bit: 6

model_name: fddnet

Number of iterations: 20
