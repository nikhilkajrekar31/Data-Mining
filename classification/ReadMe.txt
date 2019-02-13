 Select classes ABCDE from hand-written-letter data using our own data handler methods. We were asked to divide this data into training and test data. For each class using the first 30 images as training and remaining 9 images as test data. We were asked to perform classification on this data using four classifiers. 

Centroid Classification: The centroid classification calculates the centroid for each data point in the same class giving us 5 centroids as we have selected five classes from the hand-written letter data. We used the Euclidean distance to measure the distance in the centroids and the data points. The accuracy of the centroid classification algorithm on the five classes of hand-written data was 91.11% 

K Nearest Neighbors: The KNN classification algorithm compares the distance between training and testing data points using Euclidean distance as above. The value of K in KNN algorithm denotes how many close points to select to determine the classification. Here we take the value of K as 3. The elect function determines which nearest neighbor is to be selected and the point to classify to the label which most occurred. The accuracy of the KNN classification algorithm on the five classes of hand-written data was 93.33% 

Linear Regression: The linear regression classification algorithm provided by professor on the five classes of hand-written data was 88.88% 

SVM: The SVM or Support Vector Machine classification algorithm was used from the Sci-Kit library. We used linear kernel and appropriate data from hand-written letter data of five classes was selected. The predict function determines the classification of the algorithm using which we got an accuracy of 95.55%

 5-fold cross validation using all four classifiers as in task A and report accuracy and average accuracy for all classifier algorithms. We use the data handlers to split the data into testing and training data and select different dataset for each iteration. We split the data into 5 different datasets and get different accuracies for each iteration and we also calculate the average accuracy of the entire 5fold cross validation process. 

The accuracies of different classification algorithms are as follows: 

5-Fold Centroid Classification: 
Fold 1 Accuracy: 93.75% 
Fold 2 Accuracy: 97.50% 
Fold 3 Accuracy: 97.50% 
Fold 4 Accuracy: 90.00% 
Fold 5 Accuracy: 90.00% 
Average Accuracy: 93.75% 
 
5-Fold KNN Classification: 
Fold 1 Accuracy: 93.75% 
Fold 2 Accuracy: 93.75% 
Fold 3 Accuracy: 98.75% 
Fold 4 Accuracy: 97.50% 
Fold 5 Accuracy: 95.00% 
Average Accuracy: 95.75% 
 
5-Fold Linear Regression Classification: 
Fold 1 Accuracy: 96.25% 
Fold 2 Accuracy: 91.25% 
Fold 3 Accuracy: 96.25% 
Fold 4 Accuracy: 85.00% 
Fold 5 Accuracy: 87.50% 
Average Accuracy: 91.25%

5-Fold SVM Classification: 
Fold 1 Accuracy: 100.00% 
Fold 2 Accuracy: 98.75% 
Fold 3 Accuracy: 100.00% 
Fold 4 Accuracy: 96.25% 
Fold 5 Accuracy: 96.25% 
Average Accuracy: 98.25% 

