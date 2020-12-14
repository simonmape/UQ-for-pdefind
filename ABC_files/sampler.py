from numpy.random import uniform, normal
from sys import argv, stdout, stderr

# Process arguments
if len(argv) != 8:
    stderr.write("Incorrect number of arguments\n".format(argv[0]))
    exit(1)

jointMass = float(argv[1])
diffMass = float(argv[2])
udiffMass = float(argv[3])
diffMean = float(argv[4])
udiffMean = float(argv[5])
diffSD = float(argv[6])
udiffSD = float(argv[7])

totalMass = jointMass + diffMass + udiffMass
t = uniform(low=0,high=1)

diff_sampled = normal(loc=diffMean,scale=diffSD)
udiff_sampled = normal(loc=udiffMean,scale=udiffSD)

if t<diffMass/totalMass: #draw from just diffusion coefficient
    udiff_sampled =0
    
elif t<(diffMass+udiffMass)/totalMass: #draw from just u*diffusion coefficient
    diff_sampled = 0

# Print sampled parameters
stdout.write("{} {}\n".format(diff_sampled, udiff_sampled))
