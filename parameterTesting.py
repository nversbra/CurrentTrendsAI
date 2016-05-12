import csv
import os
import random
import numpy as np
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

n_readouts = []
n_components = []
damping = []
weight_scaling = []
discard_steps = []
alpha = []
lengthPenalty = []
random_seed = []


iterArray = np.arange(0, 10, 1)
for i in iterArray:

    n_components.append(random.randrange(10, 200, 1))
    damping.append(random.uniform(0, 1))
    weight_scaling.append(random.uniform(0, 2))
    n_readouts.append(random.randrange(2, 10, 1))
    discard_steps.append(random.randrange(10, 55, 1))
    alpha.append(random.uniform(0, 0.1))
    lengthPenalty.append(random.uniform(0, 0.1))
    random_seed.append(random.randrange(1, 50000, 5))


testdata=[]
test = open('test-file.csv', 'rb')
treader = csv.reader(test, delimiter=',')
for testd in treader:
    testdata.append(testd[0])
test.close()

pool = ThreadPool(4)


parameters=open('parameters.txt', 'rb')
reader = csv.reader(parameters, delimiter=',')

result = open('result.txt', 'wb')
mywriter= csv.writer(result)



#for row in reader:
#    print row

def runDeepBlueNote(k):

    command = 'Python DeepBlueNote.py training-data-file-0.csv test-data-file-0.csv output-file.csv ' + str(n_components[k]) + ' ' + \
              str(damping[k]) + ' ' + str(weight_scaling[k]) + ' ' + str(n_readouts[k]) + ' ' + str(discard_steps[k]) + ' ' + str(alpha[k]) + ' ' + str(lengthPenalty[k]) + ' ' + str(random_seed[k])
    os.system(command)

    pred = open('output-file.csv', 'rb')
    preader = csv.reader(pred, delimiter=',')
    j = 0
    accuracy = 0
    for prediction in preader:
        if testdata[j] != prediction[0]:
            accuracy += 1
        j += 1

    pred.close()

    mywriter.writerow([accuracy, n_components[i], damping[i], weight_scaling[i], n_readouts[i], discard_steps[i], alpha[i], lengthPenalty[i]])

results = pool.map(runDeepBlueNote, iterArray)

result.close()