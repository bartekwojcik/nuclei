import numpy
import pandas
from keras.models import Sequential
from keras.layers import Input, Dense
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.layers.merge import concatenate
from keras.models import Model

# load dataset
dataframe = pandas.read_csv("inputs.csv", delim_whitespace=True,header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:2]
Y = dataset[:,3:]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123)
y_train_L, y_train_R = y_train[:,0], y_train[:,1:]
y_train_L = y_train_L.reshape(-1,1)
scalarX, scalarY_L, scalarY_R = MinMaxScaler(), MinMaxScaler(), MinMaxScaler()
scalarX.fit(x_train)
scalarY_L.fit(y_train_L)
scalarY_R.fit(y_train_R)

x_train = scalarX.transform(x_train)
y_train_L = scalarY_L.transform(y_train_L)
y_train_R = scalarY_R.transform(y_train_R)
# define and fit the final model

inputs = Input(shape=(x_train.shape[1],))
first =Dense(46, activation='relu')(inputs)

#last
layer45 = Dense(45, activation='linear')(first)
layer1 = Dense(1, activation='tanh')(first)
out = [layer1,layer45]
#end last

model = Model(inputs=inputs,outputs=out)
model.compile(loss=['binary_crossentropy','mean_squared_error'], optimizer='adam')
model.fit(x_train, [y_train_L,y_train_R], epochs=1000, verbose=1)


Xnew = scalarX.transform(x_test)
y_test_L, y_test_R = y_test[:,0], y_test[:,1:]
y_test_L = y_test_L.reshape(-1,1)
y_test_L=scalarY_L.transform(y_test_L)
y_test_R=scalarY_R.transform(y_test_R)
# make a prediction
ynew = model.predict(Xnew)
# show the inputs and predicted outputs
print("SCALED VALUES")
for i in range(len(Xnew)):
	print("X=%s\n P=%s,%s\n A=%s,%s" % (Xnew[i], ynew[0][i], ynew[1][i], y_test_L[i], y_test_R[i]))

inversed_X_test = scalarX.inverse_transform(Xnew)
inversed_Y_test_L = scalarY_L.inverse_transform(y_test_L)
inversed_Y_test_R = scalarY_R.inverse_transform(y_test_R)
inversed_y_predicted_L = scalarY_L.inverse_transform(ynew[0])
inversed_y_predicted_R = scalarY_R.inverse_transform(ynew[1])
print("REAL VALUES")
for i in range(len(inversed_X_test)):
	print("X=%s\n P=%s,%s\n A=%s,%s" % (inversed_X_test[i], inversed_y_predicted_L[i],inversed_y_predicted_R[i], inversed_Y_test_L[i],inversed_Y_test_R[i]))