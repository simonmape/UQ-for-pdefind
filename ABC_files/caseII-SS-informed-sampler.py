from numpy.random import uniform, normal, gamma
from numpy import sqrt
from sys import argv, stdout, stderr

# Process arguments
if len(argv) != 2:
    stderr.write("Incorrect number of arguments\n".format(argv[0]))
    exit(1)

a_diff = float(argv[1])
ux_sigma2 = 1.0/gamma(3,5000)
diff_sigma2 = 1.0/gamma(3,78)

diffDraw = uniform(low=0,high=1)

ux_sampled = normal(loc = -0.03,scale=sqrt(ux_sigma2))

if diffDraw < a_diff:
    diff_sampled = normal(loc = 0.25,scale=sqrt(diff_sigma2))
else: 
    diff_sampled = 0
    
# Print sampled parameters
stdout.write("{:0.6f} {:0.6f} {:0.6f} {:0.6f}\n".format(ux_sampled, diff_sampled, ux_sigma2, diff_sigma2))
