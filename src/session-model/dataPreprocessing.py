import numpy as np
import pandas as pd
import datetime

def readDataIntoPanda(filename):
    data = pd.read_csv(filename, sep=',', header=None, usecols=[0,1,2], dtype={0:np.int32, 1:str, 2:np.int64})
    data.columns = ['SessionID', 'Time', 'ItemID']
    data['Time'] = data.Time.apply(lambda x : datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp())
    return data

trainfile = "../../data/yoochoose-clicks.dat"
#testfile = "../../data/yoochoose-test.dat"

train = readDataIntoPanda(trainfile)
#test = readDataIntoPanda(testfile)

print(train.describe())

#remove items appears less than 10 times
sessionsPerItem = train.groupby('ItemID').size()
train = train[np.in1d(train.ItemID, sessionsPerItem[sessionsPerItem >= 10].index)]

trainSessionLength = train.groupby('SessionID').size()
train = train[np.in1d(train.SessionID, trainSessionLength[trainSessionLength > 1].index)]


timeInterval = 12*60*60

def separateDataByTail(tailInterval, data):
    maxTime = data.Time.max()
    timeMaxPerSession = data.groupby('SessionID').Time.max()
    sessionsForTrainIndex = timeMaxPerSession[timeMaxPerSession < (maxTime - tailInterval)].index 
    sessionsForTestIndex = timeMaxPerSession[timeMaxPerSession >= (maxTime - tailInterval)].index 

    trainData = data[np.in1d(data.SessionID, sessionsForTrainIndex)]
    testData  = data[np.in1d(data.SessionID, sessionsForTestIndex)]
    return trainData, testData

tnVData, testData = separateDataByTail(timeInterval, train)

trainData, validateData = separateDataByTail(timeInterval, tnVData)

def saveFile(name, data):
  print(name, '  has ', len(data), 'Events ', data.SessionID.nunique(), 'Sessions, and', data.ItemID.nunique(), 'Items\n\n')
  data.to_csv('../../data/session-model/'+name+'.txt', sep=',', index=False)
saveFile('train', trainData)
saveFile('validation', validateData)
saveFile('test', testData)
