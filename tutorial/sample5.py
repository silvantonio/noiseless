import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from tutorial.common.dataset_handler import DatasetHandler as DH

# variables
delimiter = ","

# training data
features = pd.read_csv('data/listings/v2/features_names.csv', sep=delimiter, header=None)
X = np.loadtxt('data/listings/v2/listings_improved.csv', delimiter=delimiter)
y = np.loadtxt('data/listings/v2/listings_target.csv', delimiter=delimiter)

# Create and train
dh = DH(X=X, y=y)
dh.set_features(features)
dh.train()

# test1
X1 = np.loadtxt('data/listings/v2/test1/listings_test.csv', delimiter=delimiter)
y1 = np.loadtxt('data/listings/v2/test1/listings_test_target.csv', delimiter=delimiter)
dh.predict(X1, y1)
dh.print_report()

# test2
X1 = np.loadtxt('data/listings/v2/test2/listings_test.csv', delimiter=delimiter)
y1 = np.loadtxt('data/listings/v2/test2/listings_test_target.csv', delimiter=delimiter)
dh.predict(X1, y1)
dh.print_report()

# test3
X1 = np.loadtxt('data/listings/v2/test3/listings_test.csv', delimiter=delimiter)
y1 = np.loadtxt('data/listings/v2/test3/listings_test_target.csv', delimiter=delimiter)
dh.predict(X1, y1)
dh.print_report()

