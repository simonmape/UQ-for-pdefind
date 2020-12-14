import numpy as np
from sys import argv, stdin, stdout, stderr
import pde
from pde import CartesianGrid, MemoryStorage, PDEBase, ScalarField, plot_kymograph
import os
import tempfile
import numpy.linalg as npl
from pdefind import *

#Define the numerical integration scheme
class libraryPDE(PDEBase):
    
    def __init__(self, diff,udiff):
        self.diff = diff
        self.udiff = udiff
        
    def evolution_rate(self, state, t=0):
        assert state.grid.dim == 1
        s = "natural"
        grad_x = state.gradient(s)[0]
        c1= self.diff
        c2= self.udiff
        return c1 * state.laplace(s) + c2 * state * state.laplace(s)

   
def integrate(tfinal,diff,udiff,IC):
    
    grid = CartesianGrid([[0, 200]], 200)
    field = ScalarField(grid, IC)  
    storage = MemoryStorage()
    eq = libraryPDE(diff,udiff) #define PDE with custom parameters
    return eq.solve(field, t_range=tfinal, dt =0.1,tracker=storage.tracker(0.1)).data

# Process arguments
if len(argv) != 2:
    stderr.write("Incorrect number of arguments\n".format(argv[0]))
    exit(1)

datafile = str(argv[1])

# Read epsilon and parameters from stdin
stderr.write("Enter epsilon\n");
epsilon = int(stdin.readline())

stderr.write("Enter diff, udiff\n");
diff, udiff = [ float(num) for num in stdin.readline().split() ]

# Run simulation
I= np.zeros(200) #set the initial condition
I[95:105] = 1.0 

simulated50 = integrate(50,diff,udiff,I)
simulated100 = integrate(50,diff,udiff,simulated50)
simulated150 = integrate(50,diff,udiff,simulated100)
simulated200 = integrate(50,diff,udiff,simulated150)
simulated250 = integrate(50,diff,udiff,simulated200)

# Read datafile for observed number of heads
observed = np.loadtxt(datafile)

#Calculate error
err1 = npl.norm(simulated50-observed[1,:])
err2 = npl.norm(simulated100-observed[2,:])
err3 = npl.norm(simulated150-observed[3,:])
err4 = npl.norm(simulated200-observed[4,:])
err5 = npl.norm(simulated250-observed[5,:])
err = err1+err2+err3+err4+err5

# Print some information
stderr.write("Observed error: {}\n".format(err))

# If L2 norm is less or equal than epsilon, accept, else reject.
if err <= epsilon:
    stdout.write("accept\n")
else:
    stdout.write("reject\n")
