import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# advertType
# 1 - Unhandled
# 2 - OK
# 3 - Relevant (escalated or infriging)

# n_neighbors value
k = 1
delimiter = ","

# load features names
names = pd.read_csv('data/listings/v2/features_names.csv', sep=delimiter, header=None)
#print(names.values)

# load data
X1 = np.loadtxt('data/listings/v2/listings.csv', delimiter=delimiter)
y1 = np.loadtxt('data/listings/v2/listings_target.csv', delimiter=delimiter)

# instance of knn
knn = KNeighborsClassifier(n_neighbors=k)

# fit data to knn
knn.fit(X1, y1)

X2 = np.loadtxt('data/listings/v2/listings_test.csv', delimiter=delimiter)
y2 = np.loadtxt('data/listings/v2/listings_test_target.csv', delimiter=delimiter)

# predict
prediction = knn.predict(X2)
#print(prediction)

wrong = 0
for index in range(len(prediction)):
    #print(y2[index])
    #print(prediction[index])
    #print(y2[index]==prediction[index])
    if y2[index] != prediction[index]:
        wrong += 1

accuracy = (len(prediction)-wrong)/len(prediction)
print('Accuracy ' + str(accuracy*100) + '%')
