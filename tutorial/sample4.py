import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import csv
import pandas as pd

# advertType
# 1 - Unhandled
# 2 - OK
# 3 - Relevant (escalated or infriging)

# n_neighbors value
k = 1
delimiter = ","

# load features names
names = pd.read_csv('data/listings/v1/listings_feature_names.data', sep=delimiter, header=None)
print(names.values)

# load data
X = np.loadtxt('data/listings/v1/listings.data', delimiter=delimiter)
y = np.loadtxt('data/listings/v1/listings_target.data', delimiter=delimiter)

# instance of knn

knn = KNeighborsClassifier(n_neighbors=k)
# fit data to knn
knn.fit(X, y)

# predict
a = knn.predict(
    [
        [1, 1, 1, 3, 276, 8, 276, 1, 54.9, 4, 0, 0],
        [1, 2, 2, 1, 840, 9, 840, 1, 130.02, 14, 0, 1]
    ]
)
print(a)
