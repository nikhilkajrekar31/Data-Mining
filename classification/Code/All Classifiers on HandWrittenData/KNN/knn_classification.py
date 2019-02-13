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

#create object for class
dataHandle=dataHandler()
#create new dataset using pickdataclass
dataSet=dataHandle.pickDataClass('HandWrittenLetters.txt',dataHandle.letter_2_digit_convert('ABCDE'))
dataSet=dataSet.transpose()
number_per_class=39
#split data into testdata and traindata first 30 training and last 9 to be used for testing
testData,trainData=dataHandle.splitData2TestTrain(dataSet,number_per_class,'30:39')
#create training and testing data from split data
trainX=trainData[1:,:].transpose()
trainY=trainData[0,:]
testX=testData[1:,:].transpose()
testY=testData[0,:]

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
    for j in range(5):
        knnData=distanceArray[i][j]
        knnClassifierMatrix[i].append(distanceDict[i][knnData])
knnFinalClassification=[]
for elements in knnClassifierMatrix:
    knnFinalClassification.append(elect(elements))
    
#calculate error and test accuracy
error = testY - knnFinalClassification
TestingAccuracy = (1-np.nonzero(error)[0].size/len(error))*100
f=open('KNNClassificationAccuracy.txt','w')
f.write('%.4f Percent'%TestingAccuracy)
f.close()
np.savetxt('KNNClassificationOutput.txt',knnFinalClassification,fmt="%0.0f")