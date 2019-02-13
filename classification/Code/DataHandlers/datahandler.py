import numpy as np
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
    #split the data from file into testData and training data
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

#create new dataset using pickdataclass
dataSet=pickDataClass('filename.txt')
dataSet=dataSet.transpose()
number_per_class=39

#split data into testdata and traindata
testData,trainData=splitData2TestTrain(dataSet,number_per_class,'0:39')
         
def subroutine3():
    #create training and testing data from split data
    trainX=trainData[1:,:].transpose()
    trainY=trainData[0,:,None]
    testX=testData[1:,:].transpose()
    testY=testData[0,:]
        
def letter_2_digit_convert(self,string):
    #convert letters to digit
    self.arr = []
    for i in string:
        self.arr.append(ord(i)-64)
    self.arr=np.stack(self.arr)
    return(self.arr)