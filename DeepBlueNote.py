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




######################### Setup Part

possibleDurationValues = [2, 3/2, 1, 3/4, 1/2, 3/8, 1/4, 3/16, 1/8, 3/32, 1/16, 3/64, 1/32, 3/128, 1/64]

def mapDuration(noteTime):
        duration = noteTime / 12 / 64
        compareArray = np.full(15, duration)
        duration_index = np.argmin(abs(possibleDurationValues - compareArray))
        return(possibleDurationValues[duration_index])
        

### Process a CSV file
def inputSong( filename ):
        songNotes = [] #starting with list to avoid storage move
        songRhytm = []
        headerBegin = 0
        headerEnd = 6
        column_of_notes = 4
        column_of_rhythm = 1
        maxMidiNote = 127
        ifile  = open(filename, "rt")
        reader = csv.reader(ifile)
        rownum = 1
        noteOnTime=0
        for row in reader:
                if not (headerBegin <= rownum <= headerEnd) and not (len(row) < 4):
                    #last two rows of csv, marking the end, have no columns with data of the song (numcols < 4)
                    if (rownum % 2 == 1):
                        noteOnTime=int (row[column_of_rhythm])
                    if (rownum % 2 == 0):
                      songNotes.append(float(row[column_of_notes].strip())/maxMidiNote) #scale the values to range 0 to 1
                      noteTime = int(row[column_of_rhythm])-int(noteOnTime)
                      songRhytm.append(mapDuration(noteTime) / 2 ) #scale the values to range 0 to 1
                rownum += 1
        rhythm = np.asarray(songRhytm)
        songNotes = np.asarray([t - s for s, t in zip(songNotes, songNotes[1:])])
        notes = np.asarray(songNotes)
        ifile.close()
        return [notes, rhythm];

def padWithZeros(song, neededSize):
        zerosNeeded = neededSize - song.shape[0]
        zeros = np.zeros(zerosNeeded)
        return np.append(song, zeros);


training_data_file = sys.argv[1]
test_data_file = sys.argv[2]
output_file = sys.argv[3]

def readDataFile(datafile):
    overviewFile  = open(datafile, "rt")
    reader = csv.reader(overviewFile, delimiter=';')
    songsOverviewList = []
    for row in reader:
            songsOverviewList.append(row)
    del songsOverviewList[0] #remove header
    songsOverview = np.asarray(songsOverviewList)
    return songsOverview


training_data = readDataFile(training_data_file)
test_data = readDataFile(test_data_file)


maxLengthOfSong = 1954 #longest song has 1954 notes

trainSongsNotes = [] #using list to allow different song length
testSongsNotes = [] #using list to allow different song length
trainSongsRhythm = []
testSongsRhythm = []


index = 0;

for songInfo in training_data:
        songId = songInfo[0]
        filename = "songs-csv/" + str(songId) + ".csv"
        songNotesAndRhythm = inputSong(filename)
        trainSongsNotes.append([])
        trainSongsNotes[index] = songNotesAndRhythm[0]
        #trainSongsNotes[index] = padWithZeros(songNotesAndRhythm[0], maxLengthOfSong)
        trainSongsRhythm.append([])
        trainSongsRhythm[index] = songNotesAndRhythm[1]
        songInfo[0] = int(index) #changes the indices in the overview to the 0-179 indices in the array of songs
        index = index + 1

index = 0;
for songInfo in test_data:
    songId = songInfo[0]
    filename = "songs-csv/" + str(songId) + ".csv"
    songNotesAndRhythm = inputSong(filename)
    testSongsNotes.append([])
    testSongsNotes[index] = songNotesAndRhythm[0]
    testSongsRhythm.append([])
    testSongsRhythm[index] = songNotesAndRhythm[1]
    songInfo[0] = int(index)  # changes the indices in the overview to the 0-179 indices in the array of songs
    index = index + 1

# create 3D array where members of same class are grouped together, e.g. [ [ [..., 'art pepper', ..., ...], [..., 'art pepper', ..., ...] ], [ [..., 'benny carter', ..., ...], [..., 'benny carter', ..., ...] ] ]
def groupBy(datasetOverview, className):
        if (className == "composer"):
                return [list(g) for k, g in groupby(datasetOverview[datasetOverview[:,1].argsort()], lambda x:x[1])]
        elif (className == "instrument"):
                return [list(g) for k, g in groupby(datasetOverview[datasetOverview[:,3].argsort()], lambda x:x[3])]
        elif (className == "style"):
                return [list(g) for k, g in groupby(datasetOverview[datasetOverview[:,4].argsort()], lambda x:x[4])]
        elif (className == "year"):
                return [list(g) for k, g in groupby(datasetOverview[datasetOverview[:,5].argsort()], lambda x:x[5])]



####################### Reservoir Part

#fixed reservoir
n_readout =  3
discard = 40
esn = SimpleESN(n_readout, n_components = 100, damping = 0.2, weight_scaling = 0.99, random_state = 1, discard_steps = discard)


#### feed one or more songs to the reservoir and collect the echoes 

def collectEchoes( inputToReservoir ):
        echoes = esn.fit_transform(inputToReservoir)
        return preprocessing.scale(echoes);


####################### Learning Part


possibleComposers = list(np.unique(training_data[:,1]))
possibleInstruments = list(np.unique(training_data[:,3]))
possibleStyles = list(np.unique(training_data[:,4]))
possibleYears = list(np.unique(training_data[:,5]))




##def predictSignalIteratively( echoes, originalSong, currentTimestep):
##        svr = SVR(kernel='linear', C=1e3) #Support Vector Regression
##        trainedSVR = svr.fit(echoes[0:currentTimestep, ], originalSong[0:currentTimestep])
##        predictedValue = trainedSVR.predict(echoes[currentTimestep+1])
##        return predictedValue;


### perform the regression of the echoes to the target signal and save the learner (i.e., the regression coefficients)
def trainAtOnce( echoes, originalSong ):
        svr = SVR(kernel='linear', C=1e3, epsilon = 0.08) #Support Vector Regression
        trainedRegressor = svr.fit(echoes, originalSong[discard: ])
        #ridge = RidgeCV(alphas=(0.1, 0.5))
        #trainedRegressor = ridge.fit(echoes, originalSong[discard: ])
 #       rgr = LinearRegression(fit_intercept=True, normalize = True)
 #       trainedRegressor = rgr.fit(echoes, originalSong[discard: ])
        learnedSignal = trainedRegressor.predict(echoes)
        return(trainedRegressor, learnedSignal)

def compareNewSong( echoesNewSong, trainedSVR, newSong ):
        predictedSignal = trainedSVR.predict(echoesNewSong)
        err = mean_squared_error(predictedSignal, newSong[discard: ])
        return err;

def categoryWithSmallestError( datasetGroupedByCategory, errorList):
        meanErrors = []
        for category in np.arange(len(datasetGroupedByCategory)):
                meanError = 0
                numberOfEntries = 0
                for entry in datasetGroupedByCategory[category]:
                        index = int(entry[0])
                        meanError += errorList[index]
                        numberOfEntries += 1
                meanError = meanError / numberOfEntries
                meanErrors.append(meanError)
        return np.argmin(meanErrors);
                        

def classify(trainingData, errors):
        pred = []
        for s in np.arange(len(trainingData)):
                trainingData[s, 0] = s #replace the index to the correct csv file with the index of the entry in the trainingset
        composersGrouped = groupBy(trainingData, "composer")
        instrumentsGrouped = groupBy(trainingData, "instrument")
        stylesGrouped = groupBy(trainingData, "style")
        yearsGrouped = groupBy(trainingData, "year")
        
        pred.append( possibleComposers[ categoryWithSmallestError(composersGrouped, errors)] )
        pred.append(possibleInstruments[ categoryWithSmallestError(instrumentsGrouped, errors)])
        pred.append(possibleStyles[ categoryWithSmallestError(stylesGrouped, errors)] )
        pred.append(possibleYears[ categoryWithSmallestError(yearsGrouped, errors)] )
        return pred;
        
        


SVRs = []
learnedSignals = [] #becomes 2D list

possibleComposers = list(np.unique(training_data[:,1]))
numberOfComposers = 37
colorList = plt.cm.Dark2(np.linspace(0, 1, numberOfComposers))


for s in np.arange(len(training_data)):
        print(s)
        trainSong = trainSongsNotes[s]
        #trainSongRhythm = trainSongsRhythm[s]
        inputToReservoir = np.ndarray(shape=(len(trainSong),1), dtype=float, order='F') 
        inputToReservoir[:,0] = trainSong
 #       inputToReservoir[:,1] = trainSongRhythm
        echoes = collectEchoes( inputToReservoir )
        training = trainAtOnce(echoes, trainSong)
        SVRs.append(training[0])
        #print(training[0].coef_)
        learnedSignals.append([])
        learnedSignals[s] = training[1]

        if (s < 1): 
                target = training_data[s, 1]
                targetAsInteger = possibleComposers.index(target)
                xAxis = np.arange(len(trainSong)-discard)
                #print(training[1].shape)
                #print(len(xAxis))
                plt.plot(xAxis, trainSong[discard : ], color = colorList[targetAsInteger+6] )
                plt.plot(xAxis, training[1], color = colorList[targetAsInteger] )
                #plt.plot(xAxis, echoes, color = colorList[targetAsInteger] )

#plt.show()


lengthPenalty = 0.0001;
outFile = open(output_file, 'w')
for s in np.arange(len(test_data)):
#for s in np.arange(10):
        print("****")
        print(s)
        testSong = testSongsNotes[s]
 #       testSongRhythm = testSongsRhythm[s]
        inputToReservoir = np.ndarray(shape=(len(testSong),1), dtype=float, order='F') 
        inputToReservoir[:,0] = testSong
 #       inputToReservoir[:,1] = testSongRhythm
        echoesNewSong = collectEchoes( inputToReservoir )
        
        errors = []
        if (s == 0):
                xAxis = np.arange(len(testSong)-discard)
                plt.plot(xAxis, testSong[discard : ], color = 'k')
        for i in np.arange(len(training_data)):
                err = compareNewSong( echoesNewSong, SVRs[i], testSong)
                err += abs(len(testSong) - len(trainSongsNotes[s])) / maxLengthOfSong * lengthPenalty;
                #print(err)
                errors.append(err)
                if (s == 0 and 0 < i < 20): 
                        composer = training_data[i, 1]
                        composerAsInteger = possibleComposers.index(composer)
                        plt.plot(xAxis, SVRs[i].predict(echoesNewSong), color = colorList[composerAsInteger] )

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
        
#plt.show()


       





        
        
        
        



####################### Classification Part

### use the trained SVR (=signature) belonging to a certain composer/style/... to predict a signal from the echoes resulting from a new song
### compare this signal to the true signal for that composer/style/... and return the error 
def dissimilarity( echoesOfNewSong, trainedSVR, trueSignal):
        predictedSignal = trainedSVR.predict(echoesOfNewSong)
        err = mean_squared_error(trueSignal, predictedSignal)
        return err;


classIndices = [1, 3, 4, 5] # index of columns in overview file corresponding to different classes


 
           
                
                        
        
        




        
        
              
                
                
                
                





