import csv
import os
import random
import subprocess

import numpy as np
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

n_readouts = []
n_components = []
damping = []
weight_scaling = []
discard_steps = []
alpha = []
timesteps = []
random_seed = []


iterArray = np.arange(0, 100, 1)
for i in iterArray:

    n_components.append(random.randrange(10, 150, 10))
    damping.append(random.uniform(0.1, 0.99))
    weight_scaling.append(random.uniform(0.1, 0.9))
    n_readouts.append(random.randrange(1, 12, 1))
    discard_steps.append(random.randrange(0, 15, 1))
    alpha.append(random.uniform(0.05, 0.15))
    timesteps.append(random.randrange(100, 1500, 50))
    random_seed.append(random.randrange(30000, 50000, 5))

testdata=[]
test = open('test-file.csv', 'rt')
treader = csv.reader(test, delimiter=';')
for testd in treader:
    testdata.append(testd)
test.close()

print(testdata)

pool = ThreadPool(2)


#parameters=open('parameters.txt', 'rb')
#reader = csv.reader(parameters, delimiter=',')

result = open('result.txt', 'wt')
mywriter= csv.writer(result)



#for row in reader:
#    print row

def runDeepBlueNote(k):
    print (k)
#    print k + '   :   '+ str(n_components[k]) + ' ' + str(damping[k]) + ' ' + str(weight_scaling[k]) + ' ' + str(n_readouts[k]) + ' ' + str(discard_steps[k]) + ' ' + str(alpha[k]) + ' ' + str(lengthPenalty[k]) + ' ' + str(random_seed[k])
    
    command = 'Python DeepBlueNote.py parameters/train/train-'+str(k)+'.csv parameters/test/test-'+str(k)+'.csv parameters/out/out-'+str(k)+'.csv ' + str(n_components[k]) + ' ' + \
              str(damping[k]) + ' ' + str(weight_scaling[k]) + ' ' + str(n_readouts[k]) + ' ' + str(discard_steps[k]) + ' ' + str(alpha[k]) + ' ' + str(timesteps[k]) + ' ' + str(random_seed[k])
    print(command)
    
    with open(os.devnull, 'wt') as devnull:
       subprocess.check_call(command.split(' '), stdout=devnull, stderr=subprocess.STDOUT)

    #os.system(command)
    out= 'parameters/out/out-'+str(k)+'.csv'
    pred = open(out, 'rt')
    preader = csv.reader(pred, delimiter=',')
    j = 0
    accuracy = 0
    instrument = 0
    style = 0
    year = 0
    tempo=0
    for prediction in preader:
        p=prediction[0].split(';')
        #print(p)
        t=testdata[j]
        #print(t)
       # print t[0] + p[0]
        if t[1] != p[0]:
            accuracy += 1
        if t[3] != p[1]:
            instrument += 1
        if t[4] != p[2]:
            style += 1

        year += abs(int(t[5])-int(p[3]))

        j += 1

    pred.close()
    #mywriter.writerow([accuracy, n_components[k], damping[k], weight_scaling[k], n_readouts[k], discard_steps[k], alpha[k], lengthPenalty[k]])
    print( "accuraccy: "+ str([k, instrument, style, accuracy, n_components[k], damping[k], weight_scaling[k], n_readouts[k], discard_steps[k], alpha[k], timesteps[k], random_seed[k]]))
    #return str([instrument, style, accuracy])
    return str([k, instrument, style, accuracy, n_components[k], damping[k], weight_scaling[k], n_readouts[k], discard_steps[k], alpha[k], timesteps[k], random_seed[k]])
results = pool.map(runDeepBlueNote, iterArray)

result.close()

print(results)
