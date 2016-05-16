from simple_esn import SimpleESN
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from itertools import groupby
import matplotlib.pyplot as plt
from array import array
import numpy as np
import csv
import random
import sys
import heapq
import Utils
import Classify


training_data_file = sys.argv[1]
test_data_file = sys.argv[2]
output_file = sys.argv[3]
n_components = int(sys.argv[4])
damping = float(sys.argv[5])
weight_scaling = float(sys.argv[6])
n_readout = int(sys.argv[7])
discard = int(sys.argv[8])
alpha = float(sys.argv[9])
lengthPenalty = float(sys.argv[10])
random_seed = int(sys.argv[11])

training_data = Utils.readDataFile(training_data_file)
test_data = Utils.readDataFile(test_data_file)
trainData=Utils.importTrainingData(training_data)
testData= Utils.importTestData(test_data)
trainSongsNotes = trainData[0]
testSongsNotes = testData[0]

trainSongsRhythm = trainData[1]
testSongsRhythm = testData[1]
####################### Reservoir Part

#fixed reservoir
#n_readout =  3
#discard = 40
#esn = SimpleESN(n_readout, n_components = 100, damping = 0.2, weight_scaling = 0.99, random_state = 1, discard_steps = discard)

esn = SimpleESN(n_readout, n_components=n_components, damping = damping, weight_scaling = weight_scaling, random_state = random_seed, discard_steps=discard)

#### feed one or more songs to the reservoir and collect the echoes



####################### Learning Part


possibleComposers = list(np.unique(training_data[:,1]))
possibleInstruments = list(np.unique(training_data[:,3]))
possibleStyles = list(np.unique(training_data[:,4]))
possibleYears = list(np.unique(training_data[:,5]))






SVRs = []
learnedSignals = [] #becomes 2D list

possibleComposers = list(np.unique(training_data[:,1]))
numberOfComposers = 37
colorList = plt.cm.Dark2(np.linspace(0, 1, numberOfComposers))


for s in np.arange(len(training_data)):
        #print(s)
        trainSong = trainSongsNotes[s]
        #trainSongRhythm = trainSongsRhythm[s]
        inputToReservoir = np.ndarray(shape=(len(trainSong),1), dtype=float, order='F')
        inputToReservoir[:,0] = trainSong
 #       inputToReservoir[:,1] = trainSongRhythm
        echoes = Classify.collectEchoes(esn, inputToReservoir)
        training = Classify.trainAtOnce(echoes, trainSong, discard, alpha)
        SVRs.append(training[0])
        #print(training[0].coef_)
        learnedSignals.append([])
        learnedSignals[s] = training[1]

        if (s < 0):
                target = training_data[s, 1]
                targetAsInteger = possibleComposers.index(target)
                xAxis = np.arange(len(trainSong)-discard)
                #print(training[1].shape)
                #print(len(xAxis))
                #plt.plot(xAxis, trainSong[discard : ], color = colorList[targetAsInteger+6] )
                #plt.plot(xAxis, training[1], color = colorList[targetAsInteger] )
                #plt.plot(xAxis, echoes, color = colorList[targetAsInteger] )

#plt.show()



outFile = open(output_file, 'w')
for s in np.arange(len(test_data)):
#for s in np.arange(10):
        #print("****")
        #print(s)
        testSong = testSongsNotes[s]
 #       testSongRhythm = testSongsRhythm[s]
        inputToReservoir = np.ndarray(shape=(len(testSong),1), dtype=float, order='F')
        inputToReservoir[:,0] = testSong
 #       inputToReservoir[:,1] = testSongRhythm
        echoesNewSong = Classify.collectEchoes(esn, inputToReservoir )

        errors = []
        if (s < 0):
                xAxis = np.arange(len(testSong)-discard)
        for i in np.arange(len(training_data)):
                #plt.figure()
                #xAxis = np.arange(len(testSong) - discard)
                #plt.plot(xAxis, testSong[discard:], color='k')
                err = Classify.compareNewSong(echoesNewSong, SVRs[i], testSong, discard)
                err += abs(len(testSong) - len(trainSongsNotes[s])) / Utils.maxLengthOfSong * lengthPenalty;
                #print(err)
                errors.append(err)
                #if ( 0 <= i ):
                 #       composer = training_data[i, 1]
                  #      composerAsInteger = possibleComposers.index(composer)
                        #plt.plot(xAxis, np.gradient(SVRs[i].predict(echoesNewSong)), color = 'r' )
                        #plt.show()
                   #     if i>0:
                    #            plt.close(i-1)
                        #print composer

        indicesOf5best = heapq.nsmallest(5, range(len(errors)), errors.__getitem__)
        print(indicesOf5best)
        #prediction = classify(training_data, errors)
        prediction = training_data[np.argmin(errors)]
        print(prediction)
        outFile.write(prediction[1]+";" + prediction[3] + ";" + prediction[4] + ";" + prediction[5]+ ";"+ prediction[6] + "\n")
##        print(training_data[indicesOf5best[0], :])
##        print(training_data[indicesOf5best[1], :])
##        print(training_data[indicesOf5best[2], :])
##        print(training_data[indicesOf5best[3], :])
##        print(training_data[indicesOf5best[4], :])






####################### Classification Part

### use the trained SVR (=signature) belonging to a certain composer/style/... to predict a signal from the echoes resulting from a new song
### compare this signal to the true signal for that composer/style/... and return the error
def dissimilarity( echoesOfNewSong, trainedSVR, trueSignal):
        predictedSignal = trainedSVR.predict(echoesOfNewSong)
        err = mean_squared_error(trueSignal, predictedSignal)
        return err;


classIndices = [1, 3, 4, 5] # index of columns in overview file corresponding to different classes