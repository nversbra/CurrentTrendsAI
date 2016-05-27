import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn import tree

import Utils


def collectEchoes(esn, inputToReservoir):
    echoes = esn.fit_transform(inputToReservoir)
    #return preprocessing.scale(echoes);
    return echoes;



##def predictSignalIteratively( echoes, originalSong, currentTimestep):
##        svr = SVR(kernel='linear', C=1e3) #Support Vector Regression
##        trainedSVR = svr.fit(echoes[0:currentTimestep, ], originalSong[0:currentTimestep])
##        predictedValue = trainedSVR.predict(echoes[currentTimestep+1])
##        return predictedValue;


### perform the regression of the echoes to the target signal and save the learner (i.e., the regression coefficients)
def trainAtOnce( echoes, originalSong, discard, alpha):
        svr = SVR(kernel='linear', C=1e3, epsilon = alpha) #Support Vector Regression
        trainedRegressor = svr.fit(echoes, originalSong[discard: ])
        #regTree = tree.DecisionTreeRegressor(max_depth=2)
        #trainedRegressor = regTree.fit(echoes, originalSong[discard:])
#        ridge = RidgeCV(alphas=[alpha])
#        trainedRegressor = ridge.fit(echoes, originalSong[discard: ])
        #print(trainedRegressor.coef_)
 #       rgr = LinearRegression(fit_intercept=True, normalize = True)
#        trainedRegressor = rgr.fit(echoes, originalSong[discard: ])
        learnedSignal = trainedRegressor.predict(echoes)
        return(trainedRegressor, learnedSignal)

def extremaMetric(trainingSignals, testSignal):
        err = 0
        for i in len(testSignal):
                t= testSignal[i]
                a=[]
                for train in trainingSignals:
                        print(train)



def compareNewSong( echoesNewSong, trainedSVR, newSong, discard ):
        max_abs_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        predictedSignal = max_abs_scaler.fit_transform(trainedSVR.predict(echoesNewSong))
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
