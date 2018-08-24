from keras.layers import Input, Dense
from keras.models import Model, Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class BranchedModel:
    @staticmethod
    def get_model(size):
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
    def get_data(x_train,y_train):
        """
        :param X:
        :param Y:
        :return:
        x_train,
        [y_train_L, y_train_R],
        [scalarX, scalarY_L, scalarY_R]
        """

        y_train_L, y_train_R = y_train[:, -10:], y_train[:, :-10]
        # y_train_L = y_train_L.reshape(-1, 1)
        scalarX, scalarY_L, scalarY_R = MinMaxScaler(), MinMaxScaler(), MinMaxScaler()
        scalarX.fit(x_train)
        scalarY_L.fit(y_train_L)
        scalarY_R.fit(y_train_R)

        x_train = scalarX.transform(x_train)
        y_train_L = scalarY_L.transform(y_train_L)
        y_train_R = scalarY_R.transform(y_train_R)

        return x_train, [y_train_L,y_train_R], [scalarX, scalarY_L, scalarY_R]

    @staticmethod
    def predict(model,history,scalars, x_test, y_test):
        scalarY_L = scalars[1]
        scalarY_R = scalars[2]
        scalarX = scalars[0]
        Xnew = scalarX.transform(x_test)
        y_test_L, y_test_R = y_test[:, -10:], y_test[:, :-10]
        y_test_L = scalarY_L.transform(y_test_L)
        y_test_R = scalarY_R.transform(y_test_R)
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
            real = inversed_Y_test_R[i]
            predicted = inversed_y_predicted_R[i]
            div = real.copy()
            div[div == 0] = 1
            diff = (real - predicted) / div
            print("X=%s\n P=%s,%s\n A=%s,%s \n D=%s" % (
                inversed_X_test[i], inversed_y_predicted_L[i], inversed_y_predicted_R[i], inversed_Y_test_L[i],
                inversed_Y_test_R[i], diff))

        # plot losses ????????????????????????????
        plt.plot(history.history['loss'])
        plt.plot(history.history['dense_3_loss'])
        plt.plot(history.history['dense_2_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'dense 3', 'dense 2'], loc='upper left')
        plt.show()