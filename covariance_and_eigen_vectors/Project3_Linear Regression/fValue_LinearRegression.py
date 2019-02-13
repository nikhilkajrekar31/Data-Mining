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
trainData=trainData.transpose()
for i in range(100):
    trainX100.append(trainData[:,finalIndices[i]])
    
trainX100=np.stack(trainX100)

testX100=[]
testX=testX.transpose()
for i in range(100):
    testX100.append(testX[:,finalIndices[i]-1])
    
testX100=np.stack(testX100)

#print output of fvalues to file
finalFValues=[]
for i in range(0,100):
    finalFValues.append(sortedList[i])
np.savetxt('FinalFValues.txt',finalFValues,fmt="%0.0f",delimiter=',')
trainData=trainData.transpose()
trainY=trainData[None,0]
trainY=trainY.transpose()

#Linear Regression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_Y = LabelEncoder()
trainY[:,0]=labelencoder_Y.fit_transform(trainY[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
trainY=onehotencoder.fit_transform(trainY).toarray()
trainY=trainY.transpose()
A_train=np.ones((1,len(trainX100[0])))
A_test=np.ones((1,10))
Xtrain_padding = np.row_stack((trainX100,A_train))
Xtest_padding = np.row_stack((testX100,A_test))
B_padding = np.dot(np.linalg.pinv(Xtrain_padding.T), trainY.T)
Ytest_padding = np.dot(B_padding.T,Xtest_padding)
Ytest_padding_argmax = np.argmax(Ytest_padding,axis=0)+1

#print final output
np.savetxt('LinearRegressionOutput.txt',Ytest_padding_argmax,fmt="%0.0f")