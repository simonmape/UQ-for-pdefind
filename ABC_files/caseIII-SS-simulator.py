import numpy as np
from sys import argv, stdin, stdout, stderr
import pde
from pde import CartesianGrid, MemoryStorage, PDEBase, ScalarField, plot_kymograph
import os
import tempfile
import numpy.linalg as npl
from pdefind import *
import warnings
warnings.filterwarnings("ignore")

#Define the numerical integration scheme
class libraryPDE(PDEBase):
    
    def __init__(self, u,u2,udiff):
        self.u =u
        self.u2 = u2
        self.udiff = udiff
        
    def evolution_rate(self, state, t=0):
        assert state.grid.dim == 1
        s = "natural"
        grad_x = state.gradient(s)[0]
        c1= self.u
        c2= self.u2
        c3= self.udiff
        return c1 * state + c2 * (state**2) + c3 * state.laplace("natural") 
        
def integrate(tfinal,u,u2,udiff,IC):
    grid = CartesianGrid([[0, 200]], 200)
    field = ScalarField(grid, IC)  
    storage = MemoryStorage()
    eq = libraryPDE(u,u2,udiff) #define PDE with custom parameters
    return eq.solve(field, t_range=tfinal, dt =0.1,tracker=storage.tracker(0.1)).data

# Process arguments
if len(argv) != 2:
    stderr.write("Incorrect number of arguments\n".format(argv[0]))
    exit(1)

datafile = str(argv[1])

# Read epsilon and parameters from stdin
#stderr.write("Enter epsilon\n");
epsilon = float(stdin.readline())

#stderr.write("Enter ux, udiff\n");
u, u2, udiff, dummy1, dummy2, dummy3 = [ float(num) for num in stdin.readline().split() ]

# Run simulation
I= np.zeros(200) #set the initial condition
I[95:105] = 1.0 

simulated50 = integrate(50,u,u2,udiff,I)
simulated100 = integrate(50,u,u2,udiff,simulated50)
simulated150 = integrate(50,u,u2,udiff,simulated100)
simulated200 = integrate(50,u,u2,udiff,simulated150)
simulated250 = integrate(50,u,u2,udiff,simulated200)

# Read datafile for observed number of heads
observed = np.loadtxt(datafile)

#Calculate error
err1 = npl.norm(simulated50-observed[25,:])
err2 = npl.norm(simulated100-observed[50,:])
err3 = npl.norm(simulated150-observed[75,:])
err4 = npl.norm(simulated200-observed[100,:])
err5 = npl.norm(simulated250-observed[125,:])
err = err1+err2+err3+err4+err5

# If L2 norm is less or equal than epsilon, accept, else reject.
if err <= epsilon:
    stdout.write("accept\n")

else:
    stdout.write("reject\n")
