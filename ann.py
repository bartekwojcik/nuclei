import numpy
import pandas
from keras.models import Sequential
from keras.layers import Input, Dense
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.layers.merge import concatenate
from keras.models import Model
import matplotlib.pyplot as plt
from model_builder import ModelBuilder
from branched_model import BranchedModel

# load dataset
dataframe = pandas.read_csv("inputs.csv", delim_whitespace=True, header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:, 0:2]
Y = dataset[:, 3:]
Y = numpy.delete(Y,1,1)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123)

x_train, y_train_array, scalar_array = BranchedModel.get_data(x_train, y_train)
model = BranchedModel.get_model((x_train.shape[1],))

history =model.fit(x_train, y_train_array, epochs=300, verbose=1)

BranchedModel.predict(model,history,scalar_array,x_test,y_test)