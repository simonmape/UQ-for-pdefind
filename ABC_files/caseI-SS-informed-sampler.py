from numpy.random import uniform, normal, gamma
from numpy import sqrt
from sys import argv, stdout, stderr

# Process arguments
if len(argv) != 3:
    stderr.write("Incorrect number of arguments\n".format(argv[0]))
    exit(1)


a_diff = float(argv[1])
a_udiff = float(argv[2])
diff_sigma2 = 1.0/gamma(3,78)
udiff_sigma2 = 1.0/gamma(3,78)

t1 = uniform(low=0,high=1)
t2 = uniform(low=0,high=1)

diff_sampled = normal(loc=0.23,scale=sqrt(diff_sigma2))
udiff_sampled = normal(loc=0.64,scale=sqrt(udiff_sigma2))

if t1>a_diff: #draw from just diffusion coefficient
    diff_sampled =0
    
elif t2>a_udiff: #draw from just u*diffusion coefficient
    udiff_sampled = 0

# Print sampled parameters
stdout.write("{:0.6f} {:0.6f} {:0.6f} {:0.6f}\n".format(diff_sampled, udiff_sampled,diff_sigma2,udiff_sigma2))
