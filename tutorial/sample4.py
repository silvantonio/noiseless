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
X1 = np.loadtxt('data/listings/v2/listings.csv', delimiter=delimiter)
y1 = np.loadtxt('data/listings/v2/listings_target.csv', delimiter=delimiter)

# instance of knn
knn = KNeighborsClassifier(n_neighbors=n_neighbors)

# fit data to knn
knn.fit(X1, y1)

X2 = np.loadtxt('data/listings/v2/listings_test.csv', delimiter=delimiter)
y2 = np.loadtxt('data/listings/v2/listings_test_target.csv', delimiter=delimiter)

# predict
prediction = knn.predict(X2)

# check results
wrong = 0
for index in range(len(prediction)):
    if y2[index] != prediction[index]:
        wrong += 1

# Accuracy
accuracy = (len(prediction)-wrong)/len(prediction)
print('Accuracy ' + str(accuracy*100) + '%')
