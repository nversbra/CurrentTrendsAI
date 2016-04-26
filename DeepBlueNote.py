from simple_esn import SimpleESN
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
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
                                songList.append(float(row[column_of_notes].strip())/maxMidiNote)                       
                rownum += 1
        song = np.asarray(songList)
        ifile.close()
        return song;

def padWithZeros(song, neededSize):
        zerosNeeded = neededSize - song.shape[0]
        zeros = np.zeros(zerosNeeded)
        return np.append(song, zeros);


# temporary (& very ugly) fix
songsArtPepperIndices = [1, 2, 3, 4, 5]
songsBennyCarterIndices = [7, 8, 9, 10, 11]
##maxLengthOfSong = 0
##
##for i in songsArtPepperIndices:
##        filename = "songs-csv/" + str(i) + ".csv"
##        song = inputSong(filename)
##        if song.shape[0] > maxLengthOfSong:
##                maxLengthOfSong = song.shape[0]
##
##for i in songsBennyCarterIndices:
##        filename = "songs-csv/" + str(i) + ".csv"
##        song = inputSong(filename)
##        if song.shape[0] > maxLengthOfSong:
##                maxLengthOfSong = song.shape[0]
##
##
##print("longest song has " + str(maxLengthOfSong) + " notes")

maxLengthOfSong = 1000;

songsArtPepper = np.ndarray(shape=(5, maxLengthOfSong), dtype=float, order='F')
songsBennyCarter = np.ndarray(shape=(5, maxLengthOfSong), dtype=float, order='F')

for i in songsArtPepperIndices:
        filename = "songs-csv/" + str(i) + ".csv"
        song = inputSong(filename)
        songsArtPepper[i-1] = padWithZeros(song, maxLengthOfSong)

for i in songsBennyCarterIndices:
        filename = "songs-csv/" + str(i) + ".csv"
        song = inputSong(filename)
        songsBennyCarter[i-7] = padWithZeros(song, maxLengthOfSong)

signalArtPepper = np.empty(maxLengthOfSong)
signalArtPepper.fill(1) #constant signal y=0.2

signalBennyCarter = np.empty(maxLengthOfSong)
signalBennyCarter.fill(-1) #constant signal y=0.8



####################### Reservoir Part

#fixed reservoir
esn = SimpleESN(n_readout = 1)

#### feed one or more songs to the reservoir and collect the echoes 

def collectEchoes( inputSongs ):
        #create 2D array with size n_samples (=length of song) * n_features (=number of songs)
        inputToReservoir = np.transpose(inputSongs)#np.ndarray(shape=(maxLengthOfSong,n_features), dtype=float, order='F')
        print(inputToReservoir.shape)
        echoes = esn.fit_transform(inputToReservoir)
        #print(echoes)
        return echoes;


####################### Learning Part

### perform the regression of the echoes to the target signal and save the learner (i.e., the regression coefficients)
def learnSignature( echoes, composerSignal ):
        svr = SVR(kernel='linear', C=1e3, gamma=0.1) #Support Vector Regression, linear since target signal is linear 
        trainedSVR = svr.fit(echoes, composerSignal)
        return trainedSVR;



####################### Classification Part

### use the trained SVR (=signature) belonging to a certain composer/style/... to predict a signal from the echoes resulting from a new song
### compare this signal to the true signal for that composer/style/... and return the error 
def dissimilarity( echoesOfNewSong, trainedSVR, trueSignal):
        predictedSignal = trainedSVR.predict(echoesOfNewSong)
        err = mean_squared_error(trueSignal, predictedSignal)
        return err;
        
        



echoesArtPepper = collectEchoes( songsArtPepper )
signatureArtPepper = learnSignature (echoesArtPepper, signalArtPepper )

echoesBennyCarter = collectEchoes( songsBennyCarter )
signatureBennyCarter = learnSignature (echoesBennyCarter, signalBennyCarter )

newSong = inputSong("songs-csv/6.csv") #sixth song of Art Pepper (normally not in dataset)
newSong = padWithZeros(newSong, maxLengthOfSong)
songArray = np.array(newSong, ndmin=2)
echoesNewSong = collectEchoes(songArray)

print( dissimilarity(echoesNewSong, signatureArtPepper, signalArtPepper))
print( dissimilarity(echoesNewSong, signatureBennyCarter, signalBennyCarter))



#svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
#y_rbf = svr_rbf.fit(echoes, composerSignal).predict(echoes)

#xAxis = np.arange(echoes.shape[0])

#plt.plot(xAxis, composerSignal, c='k', label='composersignal')
#plt.scatter(xAxis, echoes.flatten(), c='g', label='echoes')
#plt.plot(xAxis, y_rbf, c='y', label='regression')
#plt.legend()
#plt.show()



