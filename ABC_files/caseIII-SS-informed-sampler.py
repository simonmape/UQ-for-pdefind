from numpy.random import uniform, normal, gamma
from numpy import sqrt
from sys import argv, stdout, stderr

# Process arguments
if len(argv) != 4:
    stderr.write("Incorrect number of arguments\n".format(argv[0]))
    exit(1)

a_u = float(argv[1])
a_u2 = float(argv[2])
a_diff = float(argv[3])

u_sigma2 = 1.0/gamma(3,50000)
u2_sigma2 = 1.0/gamma(3,50000)
diff_sigma2 = 1.0/gamma(3,78) 

uDraw = uniform(low=0,high=1)
u2Draw = uniform(low=0,high=1)
diffDraw = uniform(low=0,high=1)

if uDraw < a_u:
    u_sampled = normal(loc = 0.001,scale=sqrt(u_sigma2))
else: 
    u_sampled = 0
    
if u2Draw < a_u2:
    u2_sampled = normal(loc = -0.001,scale=sqrt(u2_sigma2))
else: 
    u2_sampled = 0

if diffDraw < a_diff:
    diff_sampled = normal(loc = 0.25,scale=sqrt(diff_sigma2))
else: 
    diff_sampled = 0
    
# Print sampled parameters
stdout.write("{:0.6f} {:0.6f} {:0.6f} {:0.6f} {:0.6f} {:0.6f}\n".format(u_sampled,u2_sampled, diff_sampled,u_sigma2,u2_sigma2,diff_sigma2))
