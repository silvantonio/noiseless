import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# advertType
# 1 - Unhandled
# 2 - OK
# 3 - Relevant (escalated or infriging)

# variables
n_neighbors = 1
delimiter = ","

# load features names
names = pd.read_csv('data/listings/v2/features_names.csv', sep=delimiter, header=None)

# load data
X = np.loadtxt('data/listings/v2/listings.csv', delimiter=delimiter)
#X = np.loadtxt('data/listings/v2/listings_improved.csv', delimiter=delimiter)
y = np.loadtxt('data/listings/v2/listings_target.csv', delimiter=delimiter)

# instance of knn
knn = KNeighborsClassifier(n_neighbors=n_neighbors)

# fit data to knn
knn.fit(X, y)

##########################
#
# Test1
#
##########################
print('Test 1\nDescription: 50 adverts from the initial set')

# load data for test 1
X1 = np.loadtxt('data/listings/v2/test1/listings_test.csv', delimiter=delimiter)
y1 = np.loadtxt('data/listings/v2/test1/listings_test_target.csv', delimiter=delimiter)

# predict
prediction = knn.predict(X1)

# check results
wrong = 0
for index in range(len(prediction)):
    if y1[index] != prediction[index]:
        wrong += 1
        # message = str(index)  + ':[' + str(prediction[index]) + '] [' + str(y1[index]) + ']'
        # print(message)

# Accuracy
print('result: ' + str((len(prediction)-wrong)) + ' out of ' + str(len(prediction)))
accuracy = (len(prediction)-wrong)/len(prediction)
print('accuracy: ' + str(accuracy*100) + '%\n')

##########################
#
# Test2
#
##########################
print('Test 2\nDescription: 10k+ adverts from 2017-07-15 to 2017-07-10')

#load data for test 2
X2 = np.loadtxt('data/listings/v2/test2/listings_test.csv', delimiter=delimiter)
y2 = np.loadtxt('data/listings/v2/test2/listings_test_target.csv', delimiter=delimiter)

# predict
prediction = knn.predict(X2)

# check results
wrong = 0
for index in range(len(prediction)):
    if y2[index] != prediction[index]:
        wrong += 1
        #message = str(index)  + ':[' + str(prediction[index]) + '] [' + str(y2[index]) + ']'
        #print(message)

# Accuracy
print('result: ' + str((len(prediction)-wrong)) + ' out of ' + str(len(prediction)))
accuracy = (len(prediction)-wrong)/len(prediction)
print('Test 2 accuracy ' + str(accuracy*100) + '%\n')

##########################
#
# Test3
#
##########################
print('Test 3\nDescription: same as test 2 but with type 3 sellerlabel hardcoded to "undefined"')

#load data for test 3
X3 = np.loadtxt('data/listings/v2/test3/listings_test.csv', delimiter=delimiter)
y3 = np.loadtxt('data/listings/v2/test3/listings_test_target.csv', delimiter=delimiter)

# predict
prediction = knn.predict(X3)

# check results
wrong = 0
for index in range(len(prediction)):
    if y3[index] != prediction[index]:
        wrong += 1
        #message = str(index)  + ':[' + str(prediction[index]) + '] [' + str(y3[index]) + ']'
        #print(message)

# Accuracy
print('result: ' + str((len(prediction)-wrong)) + ' out of ' + str(len(prediction)))
accuracy = (len(prediction)-wrong)/len(prediction)
print('Test 3 accuracy ' + str(accuracy*100) + '%\n')

##########################
#
# random stuff
#
##########################
prediction = knn.predict([
    [1, 1, 1, 1, 840, 0, 840, 1, 114.2, 1, 0],
    [1, 1, 1, 1, 840, 0, 840, 1, 105.41, 1, 0],
    [1, 1, 1, 1, 840, 0, 840, 1, 105.41, 1, -4000],
    [1, 3, 3, 1, 840, 0, 840, 1, 105.41, 1, 0],
    [1, 1, 1, 1, 840, 0, 840, 1, 114.2, 1, -4000],
    [1, 3, 3, 1, 840, 0, 840, 1, 105.41, 1, 0],
    [1, 1, 1, 1, 840, 0, 840, 1, 61.49, 1, 0],
    [1, 1, 1, 1, 840, 0, 840, 1, 57.09, 1, -4000],
    [1, 1, 1, 1, 840, 0, 840, 1, 105.41, 1, -8000],
    [1, 3, 3, 1, 840, 0, 840, 1, 137.04, 1, 0],
    [1, 1, 1, 1, 840, 0, 840, 1, 114.2, 1, 0],
    [1, 1, 1, 1, 840, 0, 840, 1, 114.2, 1, 0],
    [1, 1, 1, 1, 840, 0, 840, 1, 87.84, 1, 0],
    [1, 1, 1, 1, 840, 0, 840, 1, 105.41, 1, 0],
    [1, 1, 1, 1, 840, 0, 840, 1, 105.41, 1, -4000],
    [1, 1, 1, 1, 840, 0, 840, 1, 137.04, 1, -8000],
    [1, 1, 1, 1, 840, 0, 840, 1, 83.45, 1, 0],
    [1, 1, 1, 1, 840, 0, 840, 2, 166.91, 1, 0],
    [1, 1, 1, 1, 356, 0, 356, 1, 9.66, 5, -3996],
    [1, 1, 1, 1, 356, 0, 356, 1, 9.66, 6, -3996],
    [1, 2, 2, 1, 840, 0, 840, 1, 254.77, 1, -4000]
])
print(prediction)

prediction = knn.predict([
    [1, 1, 1, 1, 840, 5, 840, 1, 114.2, 1, 0],
    [1, 1, 1, 1, 840, 5, 840, 1, 105.41, 1, 0],
    [1, 1, 1, 1, 840, 5, 840, 1, 105.41, 1, -4000],
    [1, 3, 3, 1, 840, 5, 840, 1, 105.41, 1, 0],
    [1, 1, 1, 1, 840, 5, 840, 1, 114.2, 1, -4000],
    [1, 3, 3, 1, 840, 5, 840, 1, 105.41, 1, 0],
    [1, 1, 1, 1, 840, 5, 840, 1, 61.49, 1, 0],
    [1, 1, 1, 1, 840, 5, 840, 1, 57.09, 1, -4000],
    [1, 1, 1, 1, 840, 5, 840, 1, 105.41, 1, -8000],
    [1, 3, 3, 1, 840, 5, 840, 1, 137.04, 1, 0],
    [1, 1, 1, 1, 840, 5, 840, 1, 114.2, 1, 0],
    [1, 1, 1, 1, 840, 5, 840, 1, 114.2, 1, 0],
    [1, 1, 1, 1, 840, 5, 840, 1, 87.84, 1, 0],
    [1, 1, 1, 1, 840, 5, 840, 1, 105.41, 1, 0],
    [1, 1, 1, 1, 840, 5, 840, 1, 105.41, 1, -4000],
    [1, 1, 1, 1, 840, 5, 840, 1, 137.04, 1, -8000],
    [1, 1, 1, 1, 840, 5, 840, 1, 83.45, 1, 0],
    [1, 1, 1, 1, 840, 5, 840, 2, 166.91, 1, 0],
    [1, 1, 1, 1, 356, 5, 356, 1, 9.66, 5, -3996],
    [1, 1, 1, 1, 356, 5, 356, 1, 9.66, 6, -3996],
    [1, 2, 2, 1, 840, 5, 840, 1, 254.77, 1, -4000]
])
print(prediction)