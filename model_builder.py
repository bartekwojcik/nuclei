from keras.layers import Input, Dense
from keras.models import Model, Sequential
class ModelBuilder:
    @staticmethod
    def branched(size):
        inputs = Input(shape=size)
        first = Dense(31, activation='relu')(inputs)
        # last
        layer45 = Dense(20, activation='linear')(first)
        layer1 = Dense(10, activation='sigmoid')(first)
        out = [layer1, layer45]
        # end last
        model = Model(inputs=inputs, outputs=out)
        model.compile(loss=['binary_crossentropy', 'mean_squared_error'], optimizer='adam')
        return model

    @staticmethod
    def simple():
        model = Sequential()
        model.add(Dense(30, input_dim=2, kernel_initializer='normal', activation='relu'))
        model.add(Dense(30, kernel_initializer='normal'))
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    @staticmethod
    def wider_model():
        # create model
        model = Sequential()
        model.add(Dense(40, input_dim=2, kernel_initializer='normal', activation='relu'))
        model.add(Dense(30, kernel_initializer='normal'))
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model