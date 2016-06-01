import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import heapq

import Utils


def collectEchoes(esn, inputToReservoir):
    echoes = esn.fit_transform(inputToReservoir)
    return preprocessing.scale(echoes);



##def predictSignalIteratively( echoes, originalSong, currentTimestep):
##        svr = SVR(kernel='linear', C=1e3) #Support Vector Regression
##        trainedSVR = svr.fit(echoes[0:currentTimestep, ], originalSong[0:currentTimestep])
##        predictedValue = trainedSVR.predict(echoes[currentTimestep+1])
##        return predictedValue;


### perform the regression of the echoes to the target signal and save the learner (i.e., the regression coefficients)
def trainAtOnce( echoes, originalSong, discard, alpha):
        svr = SVR(kernel='linear', C=1e3, epsilon = alpha) #Support Vector Regression
        trainedRegressor = svr.fit(echoes, originalSong[discard: ])
        #ridge = RidgeCV(alphas=alpha)
        #trainedRegressor = ridge.fit(echoes, originalSong[discard: ])
 #       rgr = LinearRegression(fit_intercept=True, normalize = True)
 #       trainedRegressor = rgr.fit(echoes, originalSong[discard: ])
        learnedSignal = trainedRegressor.predict(echoes)
        return(trainedRegressor, learnedSignal)

def compareNewSong( echoesNewSong, trainedSVR, referenceSignal, discard ):
        predictedSignal = trainedSVR.predict(echoesNewSong)
        err = mean_squared_error(predictedSignal, referenceSignal[discard: ])
        return err;


def balanceSet( training_data, songsGroupedPerCategory, numberOfSongsPerCategory ):
    print( len(songsGroupedPerCategory))
    indicesOfBalancedSet = []
    indicesOfBalancedSetInTrainingSet = []
    for category in songsGroupedPerCategory:
        if ( len(category) >= numberOfSongsPerCategory):
            for i in np.arange(numberOfSongsPerCategory):
                index = int(category[i][0]) # index in total set
                indicesOfBalancedSet.append(index)
    for i in np.arange(len(training_data)):
        if (int(training_data[i][0]) in indicesOfBalancedSet):
            indicesOfBalancedSetInTrainingSet.append( i )
    return indicesOfBalancedSetInTrainingSet;



def sumErrorsPerValue( errors, values ):
    errorsPerValue = []
    uniqueValues = np.unique(values)
    for value in uniqueValues:
        errSum = 0
        numberOfValues = 0
        for s in np.arange(len(values)):
            if value == values[s]:
                numberOfValues += 1
                errSum += errors[s]
        errorsPerValue.append(errSum/numberOfValues)
    return uniqueValues[np.argmin(errorsPerValue)];
    
                
        



def majorityVote( errors, training_data ) :
    prediction = []
    mostRelatedIndices = heapq.nsmallest(5, range(len(errors)), errors.__getitem__)
    mostRelatedSongs = training_data[mostRelatedIndices, ]
    mostRelatedErrors = [errors[i] for i in mostRelatedIndices]
    for i in np.arange(8): # number of columns
        if (i == 1 or i ==3 or i == 4): # composer, instrument or style
            prediction.append(sumErrorsPerValue( mostRelatedErrors, mostRelatedSongs[:,i]))
        if (i == 5 or i == 6): # year and tempo
            prediction.append(str(np.average(mostRelatedSongs[:,i].astype(np.float))))
    return prediction;
                        
                    
            
            

##    
##
##    
##    for i in np.arange(8): # number of columns
##        if (i == 1 ): # composer
##            mostRelatedIndices = heapq.nsmallest(5, range(len(errors)), errors.__getitem__)
##            mostRelatedSongs = training_data[mostRelatedIndices, ]
##            uniqueValues, counts = np.unique(mostRelatedSongs[:,i], return_counts = True)
##            prediction.append(uniqueValues[np.argmax(counts)])
##        if (i == 3): #instrument
##            balancedErrors = [errors[i] for i in balancedInstrumentsIndices]
##            mostRelatedIndices = heapq.nsmallest(5, range(len(balancedErrors)), balancedErrors.__getitem__)
##            mostRelatedSongs = training_data[mostRelatedIndices, ]
##            print("instrument most related :")
##            print(mostRelatedSongs)
##            uniqueValues, counts = np.unique(mostRelatedSongs[:,i], return_counts = True)
##            prediction.append(uniqueValues[np.argmax(counts)])
##        if (i == 4): #style
##            balancedErrors = [errors[i] for i in balancedStylesIndices]
##            mostRelatedIndices = heapq.nsmallest(5, range(len(balancedErrors)), balancedErrors.__getitem__)
##            mostRelatedSongs = training_data[mostRelatedIndices, ]
##            print("style most related :")
##            print(mostRelatedSongs)
##            uniqueValues, counts = np.unique(mostRelatedSongs[:,i], return_counts = True)
##            prediction.append(uniqueValues[np.argmax(counts)])
##        if (i == 5 or i == 6): # year and tempo
##            mostRelatedIndices = heapq.nsmallest(5, range(len(errors)), errors.__getitem__)
##            mostRelatedSongs = training_data[mostRelatedIndices, ]
##            prediction.append(np.average(mostRelatedSongs[:,i].astype(np.float)))
## 
##    return prediction;
    

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



def classify(trainingData, errors, possibleComposers, possibleInstruments, possibleStyles, possibleYears):
        pred = []
        for s in np.arange(len(trainingData)):
                trainingData[s, 0] = s #replace the index to the correct csv file with the index of the entry in the trainingset
        composersGrouped = Utils.groupBy(trainingData, "composer")
        instrumentsGrouped = Utils.groupBy(trainingData, "instrument")
        stylesGrouped = Utils.groupBy(trainingData, "style")
        yearsGrouped = Utils.groupBy(trainingData, "year")

        pred.append( possibleComposers[ categoryWithSmallestError(composersGrouped, errors)] )
        pred.append(possibleInstruments[ categoryWithSmallestError(instrumentsGrouped, errors)])
        pred.append(possibleStyles[ categoryWithSmallestError(stylesGrouped, errors)] )
        pred.append(possibleYears[ categoryWithSmallestError(yearsGrouped, errors)] )
        return pred;
