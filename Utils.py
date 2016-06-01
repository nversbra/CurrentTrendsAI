
import numpy as np
import csv
from itertools import groupby

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
        songMelodic = []
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
                      pitch = float(row[column_of_notes].strip())
                      songNotes.append(pitch/maxMidiNote) 
                      noteTime = int(row[column_of_rhythm])-int(noteOnTime)
                      songRhytm.append(mapDuration(noteTime) )
                      melodic = [pitch] * noteTime
                      songMelodic = songMelodic + melodic
                rownum += 1
        rhythm = np.asarray(songRhytm)
        songNotes = np.asarray([t - s for s, t in zip(songNotes, songNotes[1:])])
        notes = np.asarray(songNotes)
        melodic = np.asarray(songMelodic)
        ifile.close()
        return [notes, rhythm, melodic];

def padWithZeros(song, neededSize):
        zerosNeeded = int(neededSize) - int(len(song))
        if (zerosNeeded > 0):
                zeros = np.zeros(zerosNeeded)
                return np.append(song, zeros)
        else:
                return song;




def readDataFile(datafile):
    overviewFile  = open(datafile, "rt")
    reader = csv.reader(overviewFile, delimiter=';')
    songsOverviewList = []
    for row in reader:
            songsOverviewList.append(row)
    #del songsOverviewList[0] #remove header
    songsOverview = np.asarray(songsOverviewList)
    return songsOverview



maxLengthOfSong = 1954 #longest song has 1954 notes



def importTrainingData(training_data):
    index = 0;
    trainSongsNotes = []
    trainSongsRhythm = []
    trainSongsMelodic = []
    for songInfo in training_data:
        songId = songInfo[0]
        filename = "songs-csv/" + str(songId) + ".csv"
        songNotesAndRhythm = inputSong(filename)
        trainSongsNotes.append([])
        trainSongsNotes[index] = songNotesAndRhythm[0]
        trainSongsRhythm.append([])
        trainSongsRhythm[index] = songNotesAndRhythm[1]
        trainSongsMelodic.append([])
        trainSongsMelodic[index] = songNotesAndRhythm[2]
        songInfo[0] = int(index) #changes the indices in the overview to the 0-179 indices in the array of songs
        index = index + 1
    return  [trainSongsNotes, trainSongsRhythm, trainSongsMelodic]

def importTestData(test_data):
    index = 0;
    testSongsNotes = []  # using list to allow different song length
    testSongsRhythm = []
    testSongsMelodic = []
    for songInfo in test_data:
        songId = songInfo[0]
        filename = "songs-csv/" + str(songId) + ".csv"
        songNotesAndRhythm = inputSong(filename)
        testSongsNotes.append([])
        testSongsNotes[index] = songNotesAndRhythm[0]
        testSongsRhythm.append([])
        testSongsRhythm[index] = songNotesAndRhythm[1]
        testSongsMelodic.append([])
        testSongsMelodic[index] = songNotesAndRhythm[2]
        songInfo[0] = int(index)  # changes the indices in the overview to the 0-179 indices in the array of songs
        index = index + 1
    return [testSongsNotes, testSongsRhythm, testSongsMelodic]


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

