from sklearn.neighbors import KNeighborsClassifier

class DatasetHandler(KNeighborsClassifier):
    n_neighbors = 1
    name = "Current dataset"
    description = "Currently loaded dataset"
    knn = None
    features = None
    prediction = None
    prediction_wrong = None
    prediction_right = None
    prediction_total = None

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X1 = None
        self.y1 = None

    def set_features(self, features):
        self.features = features

    def set_name(self, name):
        self.name = name

    def set_description(self, description):
        self.description = description

    def train(self, n_neighbors=None):
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        self.knn = KNeighborsClassifier(n_neighbors)
        self.knn.fit(self.X, self.y)

    def predict(self, X1, y1):
        self.X1 = X1
        self.y1 = y1
        self.prediction = self.knn.predict(self.X1)
        self.prediction_wrong = 0
        for index in range(len(self.prediction)):
            if self.y1[index] != self.prediction[index]:
                self.prediction_wrong += 1
        return self.prediction

    def get_prediction(self):
        return self.prediction

    def get_prediction_total(self):
        return len(self.y1)

    def get_prediction_wrong(self):
        return self.prediction_wrong

    def get_prediction_right(self):
        return self.get_prediction_total() - self.prediction_wrong

    def get_prediction_accurancy(self):
        return (self.get_prediction_total()-self.prediction_wrong)/self.get_prediction_total()

    def print_report(self):
        print('\n###################################################')
        print('Name: ' + self.name + '\nDescription:' + self.description)
        print('---------------------------------------------------')
        print('Result: ' + str(self.get_prediction_right()) + ' out of ' + str(self.get_prediction_total()))
        print('Test Accuracy ' + str(self.get_prediction_accurancy() * 100) + '%')
        print('###################################################\n')
