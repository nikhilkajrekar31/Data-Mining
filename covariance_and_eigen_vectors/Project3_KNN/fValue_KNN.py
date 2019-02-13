# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 22:04:10 2018

@author: Aditya Bharaswadkar
@author: Toshal Tambave
"""

#import required libraries
import numpy as np
import operator

#function to calculate fvalues
def fValue_calc(averagesArray,varianceArray,instances,fAverage,YtrainArrayLen):
    num=0
    denum=0
    for i in range(len(instances)):
        num+=instances[i]*(averagesArray[i]-fAverage)**2
    num=num/(len(instances)-1)
    for j in range(len(instances)):
        denum+=(instances[j]-1)*varianceArray[j]
    denum=denum/(YtrainArrayLen-len(instances))
    return num/denum

#function to calculate variance
def fVariance(aList,YtrainIndexes):
    fsumVar=0
    favgVar=0
    for i in YtrainIndexes:
        favgVar+=aList[i]
    favgVar=favgVar/len(YtrainIndexes)
    for j in YtrainIndexes:
        fsumVar+=(aList[j]-favgVar)**2
    return fsumVar/(len(YtrainIndexes)-1)

#function to calculate averages
def fAvg(aList,YtrainIndexes):
    fsumAvg=0
    for i in YtrainIndexes:
       fsumAvg+=aList[i]
    return(fsumAvg/len(YtrainIndexes))
    
#main function
def fValue_main(aList,trainY,bList,instances,fAverage,YtrainArrayLen):
    YtrainIndexes=[]
    averagesArray=[]
    varianceArray=[]
    for i in bList:
        for j in range(len(trainY)):
            if trainY[j]==i:
                YtrainIndexes.append(j)
        averagesArray.append(fAvg(aList,YtrainIndexes))
        varianceArray.append(fVariance(aList,YtrainIndexes))
        YtrainIndexes=[]
    fValues=fValue_calc(averagesArray,varianceArray,instances,fAverage,YtrainArrayLen)
    return fValues

#load traindata from text file
trainData=np.loadtxt('GenomeTrainXY.txt',delimiter=",")

#initialize traindata and testdata
trainX=(trainData[1:,:]).transpose()
trainY=trainData[0]
testX=np.loadtxt('GenomeTestX.txt',delimiter=",")

#select instances from file
instances=[11,6,11,12]
YtrainArrayLen=len(trainY)
aList={}
for i in range(len(trainX[0])):
    avgElements=0
    for j in trainX[:,i]:
        avgElements+=j
    avgElements=avgElements/len(trainY)
    aList[i]=(fValue_main(trainX[:,i],trainY,[1,2,3,4],instances,avgElements,YtrainArrayLen))
    
#sort elements in list
sortedList = sorted(aList.items(), key=operator.itemgetter(1),reverse=True)

finalIndices=[]
for i in range(100):
    finalIndices.append((sortedList[i][0]))

#initialize training and testing data
trainX100=[]
trainData = trainData.transpose()
for i in range(100):
    trainX100.append(trainData[:,finalIndices[i]])
    
trainX100=np.stack(trainX100)

testX100=[]
testX=testX.transpose()
for i in range(100):
    testX100.append(testX[:,finalIndices[i]-1])
    
testX100=np.stack(testX100)
trainX100=trainX100.transpose()
testX100=testX100.transpose()

#print output of fvalues to file
finalFValues=[]
for i in range(0,100):
    finalFValues.append(sortedList[i])
np.savetxt('FinalFValues.txt',finalFValues,fmt="%0.0f",delimiter=',')

#knn classification
#function to calculate euclidean distance
def euc_dist(trainA,trainB):
    trainI=0
    testI=0
    dist=0
    while(trainI<len(trainA) and testI<len(trainB)):
        dist=dist+(((trainA[trainI]-trainB[testI]))**2)
        trainI+=1
        testI+=1
    dist=dist**0.5    
    dist=('%.2f' % dist)
    return float(dist)

#getting the distances using euclidean distance
distanceArray=[]
distanceDict=[]
knnClassifierMatrix=[]
for i in range(0,len(testX100)):
    knnClassifierMatrix.append([])
    distanceDict.append({})
    distanceArray.append([])
    for j in range(0,len(trainX100)):
        distanceDict[i][euc_dist(trainX100[j],testX100[i])]=trainY[j]
        distanceArray[i].append(euc_dist(trainX100[j],testX100[i]))
    distanceArray[i].sort()

#elect function to select the maximum number of occurence
def elect(matrixArray):
    elecDict={}
    for i in matrixArray:
        if i in elecDict:
            elecDict[i]+=1
        else:
            elecDict[i]=1
    for key,val in elecDict.items():
        if val == max(elecDict.values()):
            return(int(key))

#taking the k=3 for final classification array of knn and adding to final classification matrix
for i in range(len(distanceArray)):
    for j in range(3):
        knnData=distanceArray[i][j]
        knnClassifierMatrix[i].append(distanceDict[i][knnData])
knnFinalClassification=[]
for elements in knnClassifierMatrix:
    knnFinalClassification.append(elect(elements))
    
#print final classification output
np.savetxt('KNNClassificationOutput.txt',knnFinalClassification,fmt="%0.0f")