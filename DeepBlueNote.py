from simple_esn import SimpleESN
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from itertools import groupby
import matplotlib.pyplot as plt
from array import array
import numpy as np
import csv
import random
import sys





######################### Setup Part

### Process a CSV file
# currently only with notes 
def inputSong( filename ):
        songList = [] #starting with list to avoid storage move
        songRhytm=[]
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
                    #scale the values to range -1 to 1
                    if (rownum % 2 == 1):
                        noteOnTime=int (row[column_of_rhythm])
                    if (rownum % 2 == 0):
                      songList.append(float(row[column_of_notes].strip())/maxMidiNote * 2 - 1)
                      songRhytm.append(int(row[column_of_rhythm])-int(noteOnTime))
                rownum += 1
        song = np.asarray(songList)
        ifile.close()
        return song;

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
    print("number of songs: " + str(songsOverview.shape[0]))
    return songsOverview


training_data = readDataFile(training_data_file)
test_data = readDataFile(test_data_file)


maxLengthOfSong = 1954 #longest song has 1954 notes
trainsongs = np.ndarray(shape=(training_data.shape[0], maxLengthOfSong), dtype=float, order='F')
testsongs = np.ndarray(shape=(test_data.shape[0], maxLengthOfSong), dtype=float, order='F')


index = 0;
for songInfo in training_data:
        songId = songInfo[0]
        filename = "songs-csv/" + str(songId) + ".csv"
        song = padWithZeros(inputSong(filename), maxLengthOfSong)
        trainsongs[index] = song
        songInfo[0] = int(index) #changes the indices in the overview to the 0-179 indices in the array of songs
        index = index + 1

index = 0;
for songInfo in test_data:
    songId = songInfo[0]
    filename = "songs-csv/" + str(songId) + ".csv"
    song = padWithZeros(inputSong(filename), maxLengthOfSong)
    testsongs[index] = song
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
esn = SimpleESN(n_readout = 20)

#### feed one or more songs to the reservoir and collect the echoes 

def collectEchoes( inputSongs ):
        #create 2D array with size n_samples (=length of song) * n_features (=number of songs)
        inputToReservoir = np.transpose(inputSongs)#np.ndarray(shape=(maxLengthOfSong,n_features), dtype=float, order='F')
        echoes = esn.fit_transform(inputToReservoir)
        #print(echoes)
        return echoes;


####################### Learning Part

### perform the regression of the echoes to the target signal and save the learner (i.e., the regression coefficients)
def learnSignature( echoes, composerSignal ):
        svr = SVR(kernel='linear', C=1e3, gamma=0.1) #Support Vector Regression, linear since target signal is linear 
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

def predict(testSet):
        fieldnames = ['id','Performer','Inst','Style','Year','Key']
        outputFile = open('output-file.csv', 'w')
        writer = csv.DictWriter(outputFile, fieldnames=fieldnames,  delimiter=";")
        writer.writeheader()

        possibleTargets = list() #list of possible targets (values to be predicted) per class
        for i in np.arange(4):
                possibleTargets.append([])
                classIndex = classIndices[i]
                possibleTargets[i] = np.unique(testSet[:,classIndex])

        for s in np.arange(len(testSet)):
                predictions = []
                for c in possibleTargets:
                        predictions.append(random.choice(c))
                id = testSet[:,0][s]
                writer.writerow({'id': id ,'Performer': predictions[0] ,'Inst': predictions[1],'Style': predictions[2],'Year': predictions[3],'Key' : "major"})

 
           
                
                        
        
        

all_indices = np.arange(180)
test_indices = all_indices[::5]
train_indices = np.delete(all_indices, test_indices)

#test = songsOverview[test_indices]
#train = songsOverview[train_indices]

trainedSVRs = []

for s in np.arange(len(training_data)):
        index = int(training_data[s, 0])
        echoes = collectEchoes( np.array(trainsongs[index], ndmin=2) )
        #print(echoes.shape)
        #print(songs[index].shape)
        
        trainedSVRs.append(learnSignature( echoes, trainsongs[index] ))
outFile = open(output_file, 'w')
for s in np.arange(len(test_data)):
        index = int(test_data[s, 0])
        echoesNewSong = collectEchoes( np.array(testsongs[index], ndmin=2) )
        errors = []
        for i in np.arange(len(training_data)):
                errors.append(dissimilarity( echoesNewSong, trainedSVRs[i], trainsongs[i] ))

        print("*********************************************")
        print("")
        print("Test song with index: " + str(test_data[s, 0]) + " is closest to the training song with index " + str(training_data[np.argmin(errors), 0]))
        print("")
        print("test song:")
        print(test_data[s])
        print("train song:")
        print(training_data[s])
        pred=training_data[s]
        outFile.write(pred[1]+";"+pred[4]+";"+pred[5]+";"+pred[6]+"\n")

#write it to outputfile
        
              
                
                
                
                


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




