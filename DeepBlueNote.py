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
import os
import csv
import random
import sys
import heapq
import Utils
import Classify
import scipy.io.wavfile
from features import mfcc
from features import logfbank
import multiprocessing
from multiprocessing.dummy import Pool as Threadpool

training_data_file = sys.argv[1]
test_data_file = sys.argv[2]
output_file = sys.argv[3]
if len(sys.argv) > 5:
        n_components = int(sys.argv[4])
        damping = float(sys.argv[5])
        weight_scaling = float(sys.argv[6])
        n_readout = int(sys.argv[7])
        discard = int(sys.argv[8])
        alpha = float(sys.argv[9])
        timesteps = 1000#float(sys.argv[10])
        random_seed = int(sys.argv[11])
else:
        n_components = 20
        damping = float(0.6)
        weight_scaling = float(0.3)
        n_readout = 5
        discard = 11
        alpha = float(0.1)
        timesteps = 1000
        random_seed = int(31610)



print("loading data...")
training_data = Utils.readDataFile(training_data_file)
test_data = Utils.readDataFile(test_data_file)

totalSet = Utils.readDataFile("dataset-balanced.csv")
#trainData=Utils.importTrainingData(training_data)
#testData= Utils.importTestData(test_data)

#trainSongsNotes = trainData[0]
#testSongsNotes = testData[0]
#trainSongsRhythm = trainData[1]
#testSongsRhythm = testData[1]
#trainSongsMelodic = trainData[2]
#testSongsMelodic = testData[2]

####################### Reservoir Part

#fixed reservoir
#n_readout =  3
#discard = 40
#esn = SimpleESN(n_readout, n_components = 100, damping = 0.2, weight_scaling = 0.99, random_state = 1, discard_steps = discard)

esn = SimpleESN(n_readout, n_components=n_components, damping = damping, weight_scaling = weight_scaling, random_state = random_seed, discard_steps=discard)

#### feed one or more songs to the reservoir and collect the echoes



####################### Learning Part

##
##possibleComposers = list(np.unique(training_data[:,1]))
##possibleInstruments = list(np.unique(training_data[:,3]))
##possibleStyles = list(np.unique(training_data[:,4]))
##possibleYears = list(np.unique(training_data[:,5]))




SVRs = []
learnedSignals0 = [] #becomes 2D list
learnedSignals1 = []

#possibleComposers = list(np.unique(training_data[:,1]))
numberOfComposers = 37
colorList = plt.cm.Dark2(np.linspace(0, 1, numberOfComposers))

pool=Threadpool(multiprocessing.cpu_count())

print("training phase...")



instrumentsGrouped = Utils.groupBy(training_data, "instrument")
balancedIndicesInstruments = Classify.balanceSet( training_data, instrumentsGrouped, 5 )
stylesGrouped = Utils.groupBy(training_data, "style")
balancedIndicesStyles =  Classify.balanceSet( training_data, stylesGrouped, 5 ) 



numcep_ = 1

#learnedSignals = np.empty(shape=[len(training_data), timesteps, numcep_])
trained_experts = []
learned_signals = []

#index = 0

iterArr=np.arange(len(training_data))

for i in iterArr:
        learned_signals.append([])
        trained_experts.append([])

#for song in training_data :

def handleTrainSongs(index):
        song = training_data[index]
        s = int(song[0])
        print(s)

        name = "songs_midi_indexed/" + str(s)
        name += '.midi.wav'

        (rate, sig) = scipy.io.wavfile.read(name)
        print(len(sig))
        mfcc_feat = mfcc(sig, rate, numcep=numcep_, winlen=0.20, winstep=0.10)
        print("mfcc calculated")
        
        max_abs_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        mfcc_f = mfcc_feat[:,0]
        inputMFCC = np.empty(shape=[mfcc_f.shape[0], 1])
        if (s==272):
                print(inputMFCC)
        inputMFCC[:,0] = max_abs_scaler.fit_transform(mfcc_f)
        if (s==272):
                print(inputMFCC)
        c=Classify
        esn = SimpleESN(n_readout, n_components=n_components, damping=damping, weight_scaling=weight_scaling,
                        random_state=random_seed, discard_steps=discard)
        echoes = max_abs_scaler.fit_transform(c.collectEchoes(esn, inputMFCC))
        
        training = c.trainAtOnce(echoes, inputMFCC[:,0], discard, alpha)
        print("model trained")
        expert = training[0]
        signal = training[1]
        signal = Utils.padWithZeros(signal, timesteps)
        signal = signal[0:timesteps]
        #trained_experts.append([])
        trained_experts[index] = expert
        #learned_signals.append([])
        learned_signals[index]=signal
        index += 1
pool.map(handleTrainSongs, iterArr)



##        for i in np.arange(numcep_):
##                mfcc_f = mfcc_feat[:,i]
##                inputMFCC = np.empty(shape=[mfcc_f.shape[0], 1])
##                print(mfcc_f.shape[0])
##                inputMFCC[:,0] = max_abs_scaler.fit_transform(mfcc_f)
## #               inputMFCC[:,1]= 2 * np.random.random_sample(mfcc_f.shape[0]) - 1
##                echoes = max_abs_scaler.fit_transform(Classify.collectEchoes(esn, inputMFCC))
##                training = Classify.trainAtOnce(echoes, inputMFCC[:,0], discard, alpha)[1]
##                training = Utils.padWithZeros(training, timesteps)
##                training = training[0:timesteps, ]
##                learnedSignals[index,:,i] = training
     
 
                
        
##        plt.figure()
##        xAxis = np.arange(mfcc_feat.shape[0])
##        plt.plot(xAxis, inputMFCC[:,0], color='k')
##        xAxis = np.arange(len(training[1][0:timesteps]))
##        plt.plot(xAxis, training[1][0:timesteps], color = colorList[index] )
##
##plt.show()


print("testing phase...")
outFile = open(output_file, 'w')

for song in test_data:
        s = int(song[0])
        print("****")
        print(s)

        predictedSignals = np.empty(shape=[timesteps, numcep_])
        max_abs_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        name = "songs_midi_indexed/" + str(s)
        name += '.midi.wav'
        (rate, sig) = scipy.io.wavfile.read(name)
        mfcc_feat = mfcc(sig, rate, numcep=numcep_, winlen=0.20, winstep=0.10)
        mfcc_f = mfcc_feat[:,0]
        inputMFCC = np.empty(shape=[mfcc_f.shape[0], 1])
        inputMFCC[:,0] = max_abs_scaler.fit_transform(mfcc_f)
        echoesNewSong = max_abs_scaler.fit_transform(Classify.collectEchoes(esn, inputMFCC))
        errors = []
        for i in np.arange(len(training_data)):
#               predictedSignal = Classify.trainAtOnce(echoesNewSong, inputMFCC[:,0], discard, alpha)[1]
#                predictedSignal = Utils.padWithZeros(predictedSignal, timesteps)
#                predictedSignal = predictedSignal[0:timesteps]
                predictedSignal = trained_experts[i].predict(echoesNewSong)
                predictedSignal = Utils.padWithZeros(predictedSignal, timesteps)
                predictedSignal = predictedSignal[0:timesteps]
                err = mean_squared_error(predictedSignal, learned_signals[i])
        #               err = Classify.compareNewSong(echoesNewSong, trained_experts[i], learned_signals[i], discard)
                errors.append(err)
                        


#        xAxis = np.arange(timesteps)
#        plt.plot(xAxis, inputMFCC[0:timesteps], color = 'k')
#        plt.plot(xAxis, predictedSignal[0:timesteps], color = 'r')
       

##        errors = np.empty(shape=[len(training_data), numcep_])
##        for n in np.arange(numcep_):
##                for i in np.arange(index):
##                        err = mean_squared_error(learnedSignals[i,:,n], predictedSignals[:,n])
##                        errors[i, n] = err
## 


        for entry in totalSet[1:,]:
                if int(entry[0]) == s:
                        print(entry)
        
        prediction = training_data[np.argmin(errors)]
        print(prediction)
        print("majority vote: ")
        prediction = Classify.majorityVote(errors, training_data)
        print(prediction)
        outFile.write(prediction[0]+";" + prediction[1] + ";" + prediction[2] + ";" + prediction[3]+ ";"+ prediction[4] + "\n")
##
##        print("most related: ")
##        print(training_data[indicesOf5best[0], :])
##        print(training_data[indicesOf5best[1], :])
##        print(training_data[indicesOf5best[2], :])
##        print(training_data[indicesOf5best[3], :])
##        print(training_data[indicesOf5best[4], :])


        #print("majority vote: ")
        #prediction = Classify.majorityVote(errors, training_data)
        
        #print(prediction)
 #       outFile.write(prediction[0]+";" + prediction[1] + ";" + prediction[2] + ";" + str(prediction[3])+ ";"+ str(prediction[4]) + "\n")







