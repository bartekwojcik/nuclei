from keras.layers import Input, Dense
from keras.models import Model, Sequential
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy


class SimpleModel:

    def simple(self):
        model = Sequential()
        model.add(Dense(30, input_dim=2, kernel_initializer='normal', activation='relu'))
        model.add(Dense(30, kernel_initializer='normal'))
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def wider_model(self):
        # create model
        model = Sequential()
        model.add(Dense(40, input_dim=2, kernel_initializer='normal', activation='relu'))
        model.add(Dense(30, kernel_initializer='normal'))
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def larger_model(self):
        # create model
        model = Sequential()
        model.add(Dense(2, input_dim=2, kernel_initializer='normal', activation='relu'))
        model.add(Dense(10, kernel_initializer='normal', activation='relu'))
        model.add(Dense(20, kernel_initializer='normal'))
        model.add(Dense(30, kernel_initializer='normal'))
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def get_data(self, x_train, y_train):
        scalarX, scalarY = StandardScaler(), StandardScaler()
        scalarX.fit(x_train)
        scalarY.fit(y_train)

        x_train = scalarX.transform(x_train)
        y_train = scalarY.transform(y_train)

        return x_train, y_train, scalarX, scalarY


    def evaluate_simple(self, x_train, y_train):

        seed = 7
        # evaluate model with standardized data`set
        numpy.random.seed(seed)
        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('mlp', KerasRegressor(build_fn=self.wider_model, epochs=50, batch_size=5, verbose=1)))
        pipeline = Pipeline(estimators)
        kfold = KFold(n_splits=10, random_state=seed)
        results = cross_val_score(pipeline, x_train, y_train, cv=kfold)
        print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
