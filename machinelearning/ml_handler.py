import numpy as np
import pandas as pd
from machinelearning.common.dataset_handler import DatasetHandler


class GeneralHandler:
    delimiter = ","
    knn = None

    def __init__(self):
        features = pd.read_csv('data/listings/features_names.csv', sep=self.delimiter, header=None)
        X = np.loadtxt('./data/listings/listings.csv', delimiter=self.delimiter)
        y = np.loadtxt('./data/listings/listings_target.csv', delimiter=self.delimiter)
        #print(X)
        #print(y)
        #print('ola2')
        # Create and train
        self.knn = DatasetHandler(X=X, y=y)
        self.knn.set_features(features)
        self.knn.train()

    def load_csv(self, txt):
        return np.loadtxt(txt, delimiter=self.delimiter)
