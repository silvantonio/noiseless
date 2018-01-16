# Load libraries
import pandas

# Load dataset
filepath = "data/iris/iris.data"
fields = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(filepath, names=fields)
print(dataset)