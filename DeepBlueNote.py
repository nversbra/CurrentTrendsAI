from simple_esn import SimpleESN
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from itertools import groupby
import matplotlib.pyplot as plt
from array import array
import numpy as np
import csv
import random
import sys





######################### Setup Part

possibleDurationValues = [2, 3/2, 1, 3/4, 1/2, 3/8, 1/4, 3/16, 1/8, 3/32, 1/16, 3/64, 1/32, 3/128, 1/64]

def mapDuration(noteTime):
        duration = noteTime / 12 / 64
        compareArray = np.full(15, duration)
        duration_index = np.argmin(abs(possibleDurationValues - compareArray))
        return(possibleDurationValues[duration_index])
        

### Process a CSV file
# currently only with notes 
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
    #print("number of songs: " + str(songsOverview.shape[0]))
    return songsOverview


training_data = readDataFile(training_data_file)
test_data = readDataFile(test_data_file)


##maxLengthOfSong = 1954 #longest song has 1954 notes
##trainSongsNotes = np.ndarray(shape=(training_data.shape[0], maxLengthOfSong), dtype=float, order='F')
##trainSongsRhythm = np.ndarray(shape=(training_data.shape[0], maxLengthOfSong), dtype=float, order='F')
##testSongsNotes = np.ndarray(shape=(test_data.shape[0], maxLengthOfSong), dtype=float, order='F')
##testSongsRhythm = np.ndarray(shape=(test_data.shape[0], maxLengthOfSong), dtype=float, order='F')


trainSongsNotes = [] #using list to allow different song length
testSongsNotes = [] #using list to allow different song length

index = 0;

for songInfo in training_data:
        songId = songInfo[0]
        filename = "songs-csv/" + str(songId) + ".csv"
        songNotesAndRhythm = inputSong(filename)
        trainSongsNotes.append([])
        trainSongsNotes[index] = songNotesAndRhythm[0]
        #trainSongsNotes[index] = padWithZeros(songNotesAndRhythm[0], maxLengthOfSong)
  #      trainSongsRhythm[index] = padWithZeros(songNotesAndRhythm[1], maxLengthOfSong)
        songInfo[0] = int(index) #changes the indices in the overview to the 0-179 indices in the array of songs
        index = index + 1

index = 0;
for songInfo in test_data:
    songId = songInfo[0]
    filename = "songs-csv/" + str(songId) + ".csv"
    songNotesAndRhythm = inputSong(filename)
    testSongsNotes.append([])
    testSongsNotes[index] = songNotesAndRhythm[0]
#    testSongsRhythm[index] = padWithZeros(songNotesAndRhythm[1], maxLengthOfSong)
    songInfo[0] = int(index)  # changes the indices in the overview to the 0-179 indices in the array of songs
    index = index + 1

# create 3D array where members of same class are grouped together, e.g. [ [ [..., 'art pepper', ..., ...], [..., 'art pepper', ..., ...] ], [ [..., 'benny carter', ..., ...], [..., 'benny carter', ..., ...] ] ]
def groupBy(datasetOverview, className):
        if (className == "composer"):
                return [list(g) for k, g in groupby(datasetOverview, lambda x:x[1])]
        elif (className == "instrument"):
                return [list(g) for k, g in groupby(datasetOverview, lambda x:x[3])]
        elif (className == "style"):
                return [list(g) for k, g in groupby(datasetOverview, lambda x:x[4])]
        elif (className == "year"):
                return [list(g) for k, g in groupby(datasetOverview, lambda x:x[5])]



####################### Reservoir Part

#fixed reservoir
n_readout =  1
discard = 10
esn = SimpleESN(n_readout, n_components = 4, damping = 0.5, weight_scaling = 0.7, random_state = 1, discard_steps = discard)


#### feed one or more songs to the reservoir and collect the echoes 

def collectEchoes( inputToReservoir ):
        #create 2D array with size n_samples (=length of song) * n_features (=number of songs)
        #inputToReservoir = np.transpose(inputSongs)#np.ndarray(shape=(maxLengthOfSong,n_features), dtype=float, order='F')
        echoes = esn.fit_transform(inputToReservoir)
        #print(np.amax(echoes))
        #print(echoes)
        return echoes;


####################### Learning Part

### perform the regression of the echoes to the target signal and save the learner (i.e., the regression coefficients)
def learnSignature( echoes, composerSignal ):
        svr = SVR(kernel='rbf', C=1e3, gamma=0.1) #Support Vector Regression 
        trainedSVR = svr.fit(echoes, composerSignal)
        #print(trainedSVR.coef_)
        return trainedSVR;



####################### Classification Part

### use the trained SVR (=signature) belonging to a certain composer/style/... to predict a signal from the echoes resulting from a new song
### compare this signal to the true signal for that composer/style/... and return the error 
def dissimilarity( echoesOfNewSong, trainedSVR, trueSignal):
        predictedSignal = trainedSVR.predict(echoesOfNewSong)
        err = mean_squared_error(trueSignal, predictedSignal)
        return err;


classIndices = [1, 3, 4, 5] # index of columns in overview file corresponding to different classes

##def predict(testSet):
##        fieldnames = ['id','Performer','Inst','Style','Year','Key']
##        outputFile = open('output-file.csv', 'w')
##        writer = csv.DictWriter(outputFile, fieldnames=fieldnames,  delimiter=";")
##        writer.writeheader()
##
##        possibleTargets = list() #list of possible targets (values to be predicted) per class
##        for i in np.arange(4):
##                possibleTargets.append([])
##                classIndex = classIndices[i]
##                possibleTargets[i] = np.unique(testSet[:,classIndex])
##
##        for s in np.arange(len(testSet)):
##                predictions = []
##                for c in possibleTargets:
##                        predictions.append(random.choice(c))
##                id = testSet[:,0][s]
##                writer.writerow({'id': id ,'Performer': predictions[0] ,'Inst': predictions[1],'Style': predictions[2],'Year': predictions[3],'Key' : "major"})

 
           
                
                        
        
        

##all_indices = np.arange(180)
##test_indices = all_indices[::5]
##train_indices = np.delete(all_indices, test_indices)

#test = songsOverview[test_indices]
#train = songsOverview[train_indices]

#trainedSVRs = []

#allEchoes = np.ndarray(shape=(len(training_data),maxLengthOfSong,n_readout), dtype=float, order='F')

numberOfSamples = 0 #count the total number of notes
for song in np.arange(len(trainSongsNotes)):
        numberOfSamples += len(trainSongsNotes[song]) - 1 - discard


                

echoSet = np.ndarray(shape=(numberOfSamples, n_readout), dtype=float, order='F')
targets = []
possibleTargets = list(np.unique(training_data[:,1]))


print("training phase...")

numberOfComposers = 36
colorList = plt.cm.Dark2(np.linspace(0, 1, numberOfComposers))

indexInEchoSet = 0

indicesShuffled = list(range(len(training_data)))
random.shuffle(indicesShuffled)

for s in np.arange(len(training_data)):
        print(s)
        index = indicesShuffled[s]
        inputToReservoir = np.ndarray(shape=(len(trainSongsNotes[index]),1), dtype=float, order='F') 
        inputToReservoir[:,0] = trainSongsNotes[index]
        #inputToReservoir[:,1] = trainSongsRhythm[index]
        
        echoes = collectEchoes( inputToReservoir )
        print(echoes.shape)
        echoes = np.asarray([t - s for s, t in zip(echoes, echoes[1:])])
        print(echoes.shape)
        target = training_data[index, 1]
        targetAsInteger = possibleTargets.index(target)

        xAxis = np.arange(len(trainSongsNotes[index])-discard-1)
        plt.plot(xAxis, echoes[:,0].flatten(), color = colorList[targetAsInteger] )
        for echo in echoes:
                echoSet[indexInEchoSet, ] = echo
                targets.append(target)
                indexInEchoSet += 1









targets = np.asarray(targets)
##targetsAsIntegersShuffled =  np.take(targetsAsIntegers, indices)
##
##echoSetShuffled = np.take(echoSet, indices, axis = 
##
##print(echoSet.shape)
##print(targetsAsIntegersShuffled.shape)


print("number of samples: " + str(numberOfSamples))
print(echoSet.shape)
decisionTree = tree.DecisionTreeClassifier(min_samples_leaf=10)
decisionTree = decisionTree.fit(echoSet, targets)
#rf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
#decisionTree = rf.fit(echoSet, targets)
print(decisionTree.score(echoSet, targets))






print("predicting phase...")
        
outFile = open(output_file, 'w')
for s in np.arange(len(test_data)):
        print(s)
        index = int(test_data[s, 0])
        inputToReservoir = np.ndarray(shape=(len(testSongsNotes[s]),1), dtype=float, order='F') 
        inputToReservoir[:,0] = testSongsNotes[s]
        #inputToReservoir[:,1] = testSongsRhythm[s]
        echoesNewSong = collectEchoes( inputToReservoir )
        #print(echoes.shape)
        #errors = []
        #for i in np.arange(len(training_data)):
                #errors.append(dissimilarity( echoesNewSong, trainedSVRs[i], trainSongsNotes[i] ))
                #print((allEchoes[i])[:,0].shape)
                       
                #echoesNew = np.ndarray(shape=(maxLengthOfSong,1), dtype=float, order='F')
                #echoesNew[:,0] = echoesNewSong[:,0]
                #print(echoesNew.shape)
                #print(np.asarray(allEchoes[i][:,0]).shape)
                
                #trainedSVR = learnSignature(echoesNewSong, allEchoes[i][:,0])
                #errors.append(dissimilarity( echoesNewSong, trainedSVR, (allEchoes[i])[:,0] ))
        #print( np.argmin(errors) )
        #pred=training_data[np.argmin(errors), ]
        #print(pred)

        
        predictionProbabilities = np.zeros( numberOfComposers )
        for echo in echoesNewSong:
                predictionProbabilities =  predictionProbabilities + decisionTree.predict_proba(echo)

        #print([x / maxLengthOfSong for x in predictionProbabilities])
        composerIndex = np.argmax(predictionProbabilities)
        print(composerIndex)
        print("-----")
        composer = possibleTargets[composerIndex]
                
        
        outFile.write(composer+";" + ";" +";"+ ";"+"\n")

        plt.show()



        
        
              
                
                
                
                


##targetsGroupedByComposer = groupBy(train, "composer")
##xAxis = np.arange(maxLengthOfSong)
##
##
##colorIndex = 0
##
##colorList = plt.cm.Dark2(np.linspace(0, 1, 10))
##
##predict(test)

##for composer in targetsGroupedByComposer:
##        #print("**************") 
##        #print(colorIndex)
##        for songMetaData in composer:
##                i = int(songMetaData[0])
##                if (i < 36) :
##                        color = colorList[colorIndex]
##                        echoes = collectEchoes( np.array(songs[i], ndmin=2) )
##                        plt.plot(xAxis, echoes.flatten(), c=color, label=songMetaData[1])
##        colorIndex = colorIndex + 1
##
##plt.show()






#svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
#y_rbf = svr_rbf.fit(echoes, composerSignal).predict(echoes)

#xAxis = np.arange(echoes.shape[0])

#plt.plot(xAxis, composerSignal, c='k', label='composersignal')
#plt.scatter(xAxis, echoes.flatten(), c='g', label='echoes')
#plt.plot(xAxis, y_rbf, c='y', label='regression')
#plt.legend()




