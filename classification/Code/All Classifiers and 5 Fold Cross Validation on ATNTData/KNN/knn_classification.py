import numpy as np 
class dataHandler(object):
    #load file and pick data from the file
    def pickDataClass(self,filename,class_ids):
        loadfile=np.loadtxt(filename,delimiter=',')
        data=[]
        self.lenclassid=len(class_ids)
        for i in range (len(loadfile[0])):
            if loadfile[0][i] in class_ids:
                data.append(loadfile[:,i])
            else:
                continue
            rdata=np.stack(data)        
        return rdata
    
    def splitData2TestTrain(self,filename, number_per_class, testInstances):
        #split the data from file into testdata and training data
        testInstances=testInstances.split(':')
        self.start=int(testInstances[0])
        self.end=int(testInstances[1])
        testData=[]
        trainData=[]
        max=number_per_class
        min=0
        while max<=len(filename[0]):
            testData.append(filename[:,self.start:self.end])
            trainData.append(filename[:,min:self.start])
            trainData.append(filename[:,self.end:max])
            self.start+=number_per_class
            self.end+=number_per_class
            max+=number_per_class
            min+=number_per_class
        testData=np.hstack(testData)
        trainData=np.hstack(trainData)
        return(testData,trainData)
        
    def letter_2_digit_convert(self,string):
        #convert letters to digit
        self.arr = []
        for i in string:
            self.arr.append(ord(i)-64)
        self.arr=np.stack(self.arr)
        return(self.arr)
        
    def main(self,frm,to):
        testInstances=''
        testInstances=testInstances+str(frm)
        testInstances+=':'
        testInstances+=str(to)
        
        #create object for class
        dataHandle=dataHandler()
        
        #create new dataset using pickdataclass
        dataSet=dataHandle.pickDataClass('ATNTFaceImages400.txt',[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40])
        dataSet=dataSet.transpose()
        number_per_class=10
        
        #split data into testdata and traindata
        testData,trainData=dataHandle.splitData2TestTrain(dataSet,number_per_class,testInstances)
    
        #create training and testing data from split data
        trainX=trainData[1:,:].transpose()
        trainY=trainData[0,:]
        testX=testData[1:,:].transpose()
        testY=testData[0,:]
        
        #getting the distances using euclidean distance
        distanceArray=[]
        distanceDict=[]
        knnClassifierMatrix=[]
        for i in range(0,len(testX)):
            knnClassifierMatrix.append([])
            distanceDict.append({})
            distanceArray.append([])
            for j in range(0,len(trainX)):
                distanceDict[i][euc_dist(trainX[j],testX[i])]=trainY[j]#here
                distanceArray[i].append(euc_dist(trainX[j],testX[i]))
            distanceArray[i].sort()
            
        #taking the k=4 for final classification array of knn and adding to final classification matrix
        for i in range(len(distanceArray)):
            for j in range(4):
                knnData=distanceArray[i][j]
                knnClassifierMatrix[i].append(distanceDict[i][knnData])
        knnFinalClassification=[]
        for elements in knnClassifierMatrix:
            knnFinalClassification.append(elect(elements))
                    
        #calculate error and test accuracy
        error = testY - knnFinalClassification
        TestingAccuracy = (1-np.nonzero(error)[0].size/len(error))*100
        return(knnFinalClassification,TestingAccuracy)
        
def euc_dist(trainA,trainB):
    #function to calculate euclidean distance
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

dataHand=dataHandler()
j=2
finalClassifiedList=[]
finalAccuracyList=[]
for i in range(0,10,2):
    classifiedList,accuracy = dataHand.main(i,j)
    finalClassifiedList.append(classifiedList)
    finalAccuracyList.append(accuracy)
    j+=2
sumAcc=0
for i in finalAccuracyList:
    sumAcc=sumAcc+i
avgAccuracy=sumAcc/5
np.savetxt('KNNClassificationOutput.txt',finalClassifiedList,fmt="%0.0f")
np.savetxt('KNNAccuracyOutput.txt',finalAccuracyList,fmt="%0.2f")
file=open("KNNAvgAccuracyOutput.txt","w")
file.write(str(avgAccuracy))
file.close()