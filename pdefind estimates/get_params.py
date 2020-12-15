import numpy as np
from tqdm import tqdm 

def convert_to_list(mat,des_len):
    return [mat[i*500:(i+1)*500,:] for i in range(des_len)]
    
def compute_params(dens_list, dest_mat):  
    for i in tqdm(range(len(dens_list))):
        U = dens_list[i]
        dt=2
        dx =1
        Ut, R, rhs_des = build_linear_system(np.transpose(U),dt,dx,D=2,P=2,time_diff = 'poly',deg_x =4)
        w = TrainSTRidge(R,Ut,10**-2,0.01,normalize =2)
        dest_mat[:,i] = np.real(w[:,0])
    return dest_mat    
