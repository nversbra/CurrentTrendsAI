import csv
import itertools
import random

n_readouts = []
n_components = []
damping = []
weight_scaling = []
discard_steps = []
alpha = []
lengthPenalty = []

i=0
for i in range(0, 1000):

    n_components.append(random.randrange(10,200,1))
    damping.append(random.uniform(0, 1 ))
    weight_scaling.append(random.uniform(0, 2))
    n_readouts.append(random.randrange(2,10,1))
    discard_steps.append(random.randrange(10,55,1))
    alpha.append(random.uniform(0, 0.1))
    lengthPenalty.append(random.uniform(0, 0.1))
parameters = open('parameters.txt', 'wb')
mywriter= csv.writer(parameters)
for row in zip(n_components, damping,weight_scaling, n_readouts, discard_steps, alpha, lengthPenalty):
    mywriter.writerows([row])
parameters.close()