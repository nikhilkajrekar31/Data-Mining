# -*- coding: utf-8 -*-


import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.cluster import KMeans

fileLoad= np.loadtxt('HandWrittenLetters.txt',delimiter=',')
handdata=fileLoad[1:,0:]
y_true=fileLoad[0,0:]
handdata=handdata.transpose()



KM=KMeans(n_clusters=26)
y_pred=KM.fit_predict(handdata)



C = confusion_matrix(y_true, y_pred)
C = C.T

ind = linear_assignment(-C)
C_opt = C[:,ind[:,1]]
acc_opt = np.trace(C_opt)/np.sum(C_opt)
print(acc_opt)
