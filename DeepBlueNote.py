from simple_esn import SimpleESN
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from itertools import groupby
import matplotlib.pyplot as plt
from array import array
import numpy as np
import csv





######################### Setup Part

### Process a CSV file
# currently only with notes 
def inputSong( filename ):
        songList = [] #starting with list to avoid storage move 
        headerBegin = 0
        headerEnd = 6
        column_of_notes = 4
        maxMidiNote = 127
        ifile  = open(filename, "rt")
        reader = csv.reader(ifile)
        rownum = 0
        for row in reader:
                if not(headerBegin <= rownum <= headerEnd) and (rownum % 2 == 0):
                        #last two rows of csv, marking the end, have no columns with data of the song (numcols < 4)
                        if not (len(row)<4):
                                #scale the values to range -1 to 1
                                songList.append(float(row[column_of_notes].strip())/maxMidiNote * 2 - 1)                       
                rownum += 1
        song = np.asarray(songList)
        ifile.close()
        return song;

def padWithZeros(song, neededSize):
        zerosNeeded = neededSize - song.shape[0]
        zeros = np.zeros(zerosNeeded)
        return np.append(song, zeros);



overviewFile  = open("dataset-balanced.csv", "rt")
reader = csv.reader(overviewFile, delimiter=';')
songsOverviewList = []
for row in reader:
        songsOverviewList.append(row)
del songsOverviewList[0] #remove header
songsOverview = np.asarray(songsOverviewList)


maxLengthOfSong = 1953 #longest song has 1953 notes
songs = np.ndarray(shape=(181, maxLengthOfSong), dtype=float, order='F')
index = 0;
for songInfo in songsOverview:
        songId = songInfo[0]
        filename = "songs-csv/" + str(songId) + ".csv"
        song = padWithZeros(inputSong(filename), maxLengthOfSong)
        songs[index] = song
        songInfo[0] = index #changes the indices in the overview to the 0-179 indices in the array of songs
        index = index + 1



# create 3D array where members of same class are grouped together, e.g. [ [ [..., 'art pepper', ..., ...], [..., 'art pepper', ..., ...] ], [ [..., 'benny carter', ..., ...], [..., 'benny carter', ..., ...] ] ]
def groupBy(datasetOverview, className):
        if (className == "composer"):
                return [list(g) for k, g in groupby(datasetOverview, lambda x:x[1])]



####################### Reservoir Part

#fixed reservoir
esn = SimpleESN(n_readout = 1)

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
        

all_indices = np.arange(180)
test_indices = all_indices[::5]
train_indices = np.delete(all_indices, test_indices)

songs_test = songs[test_indices]
songs_train = songs[train_indices]
targets_test = songsOverview[test_indices]
targets_train = songsOverview[train_indices]


targetsGroupedByComposer = groupBy(targets_train, "composer")
xAxis = np.arange(maxLengthOfSong)

print(targetsGroupedByComposer[2][1])
print(songs_train[11])

colorIndex = 0

for composer in targetsGroupedByComposer:
        color = [1, 1, colorIndex * 1/36]
        for songMetaData in composer:
                i = songMetaData[0]
                print(i)
                print(songs_train[11])
                print(songs_train[i]) ####### ??????????? when i = 11 this does not give the same result as songs_train[11]
                echoes = collectEchoes( np.array(songs_train[i], ndmin=2) )
                print(len(xAxis))
                print(len(echoes.flatten()))
                #print(songMetaData)
                #print(songs_train[index])
                plt.scatter(xAxis, echoes.flatten(), c=color, label=songMetaData[1])
        colorIndex = colorIndex + 1

plt.show()






#svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
#y_rbf = svr_rbf.fit(echoes, composerSignal).predict(echoes)

#xAxis = np.arange(echoes.shape[0])

#plt.plot(xAxis, composerSignal, c='k', label='composersignal')
#plt.scatter(xAxis, echoes.flatten(), c='g', label='echoes')
#plt.plot(xAxis, y_rbf, c='y', label='regression')
#plt.legend()




