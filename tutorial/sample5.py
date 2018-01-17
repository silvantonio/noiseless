import numpy as np
import pandas as pd
from tutorial.common.dataset_handler import DatasetHandler as DH

# advertType
# 1 - Unhandled
# 2 - OK
# 3 - Relevant (escalated or infriging)

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

# # test1
# X1 = np.loadtxt('data/listings/v2/test1/listings_test.csv', delimiter=delimiter)
# y1 = np.loadtxt('data/listings/v2/test1/listings_test_target.csv', delimiter=delimiter)
# dh.predict(X1, y1)
# dh.print_report()

# test2
X1 = np.loadtxt('data/listings/v2/test2/listings_test.csv', delimiter=delimiter)
y1 = np.loadtxt('data/listings/v2/test2/listings_test_target.csv', delimiter=delimiter)
dh.predict(X1, y1)
dh.print_report()

# test3
X1 = np.loadtxt('data/listings/v2/test3/listings_test.csv', delimiter=delimiter)
y1 = np.loadtxt('data/listings/v2/test3/listings_test_target.csv', delimiter=delimiter)
#print('Test 3\nDescription: same as test 2 but with type 3 sellerlabel hardcoded to "undefined"')
dh.predict(X1, y1)
dh.print_report()

