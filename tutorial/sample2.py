import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# n_neighbors value
k = 1

# load iris data
iris = datasets.load_iris()
#print(iris)
#print(iris.DESCR)

# Iris categories names
t = iris.target_names
#print(t)

# Iris feature names
f = iris.feature_names
#print(f)

# Iris data
X = iris.data
#print(X)
# Iris categories
y = iris.target
#print(y)

# instance of knn
knn = KNeighborsClassifier(n_neighbors=k)
#print(knn)

# fit data to knn
knn.fit(X, y)

# predict
a = knn.predict([[3, 5, 4, 2]])
print(t[a])

# predict
a = knn.predict([[4, 3, 1, 2]])
print(t[a])
