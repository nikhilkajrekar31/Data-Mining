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
        trainX=trainData[1:,:]
        trainY=trainData[0,:,None]
        testX=testData[1:,:]
        testY=testData[0,:]
        
        #import statements
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        labelencoder_Y = LabelEncoder()
        trainY[:,0]=labelencoder_Y.fit_transform(trainY[:,0])
        onehotencoder=OneHotEncoder(categorical_features=[0])
        trainY=onehotencoder.fit_transform(trainY).toarray()
        trainY=trainY.transpose()
        #linear regression
        A_train=np.ones((1,len(trainX[0])))
        A_test=np.ones((1,len(testY)))
        Xtrain_padding = np.row_stack((trainX,A_train))
        Xtest_padding = np.row_stack((testX,A_test))
        B_padding = np.dot(np.linalg.pinv(Xtrain_padding.T), trainY.T)
        Ytest_padding = np.dot(B_padding.T,Xtest_padding)
        Ytest_padding_argmax = np.argmax(Ytest_padding,axis=0)+1
        
        #calculate error and accuracy
        err_test_padding = testY - Ytest_padding_argmax
        TestingAccuracy_padding = (1-np.nonzero(err_test_padding)[0].size/len(err_test_padding))*100
        return(Ytest_padding_argmax,TestingAccuracy_padding)
        
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
np.savetxt('LinearRegressionClassificationOutput.txt',finalClassifiedList,fmt="%0.0f")
np.savetxt('LinearRegressionAccuracyOutput.txt',finalAccuracyList,fmt="%0.2f")
file=open("LinearRegressionAvgAccuracyOutput.txt","w")
file.write(str(avgAccuracy))
file.close()