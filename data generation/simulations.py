import numpy as np
import random as r
from tqdm import tqdm

def cellmover(i,j,G,P,rho_x,rho_y,Lx,Ly):
    if np.random.uniform(low=0.0, high=1.0, size=1)[0] > P:
        return G
    S = np.random.uniform(low=0.0, high=1.0, size=1)[0]
    if S < (1-rho_y)/4:
        if j == Ly-1:
            if G[(i,0)] == 0:
                G[(i,j)] =0
                G[(i,0)] = 1
            return G
        if G[(i,j+1)]==0:
            G[(i,j)] = 0
            G[(i,j+1)]=1

    elif S < 1/2:
        if j == 0:
            if G[(i,Ly-1)] == 0:
                G[(i,j)] =0
                G[(i,Ly-1)] =1
            return G
        if G[(i,j-1)] == 0:
            G[(i,j)] = 0
            G[(i,j-1)] = 1

    elif S < 1/2 + (1-rho_x)/4:
        if i == 0:
            if G[(Lx-1,j)] == 0:
                G[(i,j)] = 0
                G[(Lx-1,j)] =1
            return G
        if G[(i-1,j)]==0:
            G[(i,j)] =0
            G[(i-1,j)]=1

    else:
        if i == Lx-1:
            if G[(0,j)] == 0:
                G[(i,j)] = 0
                G[(0,j)]=1
            return G
        if G[(i+1,j)] ==0:
            G[(i,j)] = 0
            G[(i+1,j)] =1
    return G
    
def cellproliferator(i,j,G,prolif_prob,Lx,Ly):
    if np.random.uniform(low=0.0, high=1.0, size=1)[0] > prolif_prob:
        return G
    S = np.random.uniform(low=0.0, high=1.0, size=1)[0]
    if S < 0.25:
        if j == Ly-1:
            if G[(i,0)] == 0:
                G[(i,0)] = 1
            return G
        if G[(i,j+1)]==0:
            G[(i,j+1)]=1

    elif S < 0.5:
        if j == 0:
            if G[(i,Ly-1)] == 0:
                G[(i,Ly-1)] =1
            return G
        if G[(i,j-1)] == 0:
            G[(i,j-1)] = 1

    elif S < 0.75:
        if i == 0:
            if G[(Lx-1,j)] == 0:
                G[(Lx-1,j)] =1
            return G
        if G[(i-1,j)]==0:
            G[(i-1,j)]=1

    else:
        if i == Lx-1:
            if G[(0,j)] == 0:
                G[(0,j)]=1
            return G
        if G[(i+1,j)] ==0:
            G[(i+1,j)] =1
    return G          
    
def update(G,move_prob,prolif_prob,rho_x,rho_y,Lx,Ly):
    selectfrom = list(np.transpose(np.nonzero(G)))
    permutation= r.sample(selectfrom,len(selectfrom))
    
    for pe in permutation:
        (i,j) = (pe[0], pe[1])
        G = cellproliferator(i,j,G,prolif_prob,Lx,Ly)
        G = cellmover(i,j,G,move_prob,rho_x,rho_y,Lx,Ly)
    return G      

def simulate(tf, time_step, init_cond,move_prob,prolif_prob,rho_x,rho_y):
    (Lx,Ly)= (np.shape(init_cond)[0],np.shape(init_cond)[1])
    data = np.zeros((int(tf/time_step),Lx,Ly))
    data[0,:,:] = init_cond
    time = 0.0
    count = 0
    while time < tf-time_step:
        count = count +1
        time = time + time_step
        data[count,:,:] = update(data[count-1,:,:],move_prob,prolif_prob,rho_x, rho_y,Lx,Ly)
    return data
    
def average_densities(data, times):
    return [data[x,:,:].mean(1) for x in times]

def average_over_densities(data_list):
    xextent = data_list[0][0].shape[0]
    textent = len(data_list[0])
    densities = np.zeros((textent,xextent))
    for i in range(textent):
        for j in range(xextent):
            densities[i,j] = np.average([data_list[x][i][j] for x in range(len(data_list))])
    return densities

def average_over_realizations(N, times, tf, time_step, init_cond,move_prob,prolif_prob,rho_x,rho_y):
    densities = average_densities(simulate(tf, time_step, init_cond,move_prob,prolif_prob,rho_x,rho_y), times)
    for i in range(1,N):
        sim = simulate(tf, time_step, init_cond,move_prob,prolif_prob,rho_x,rho_y)
        densities = [(i*densities[x] + average_densities(sim, times)[x])/(i+1) for x in range(len(densities))]
    return densities
