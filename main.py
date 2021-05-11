from tensorflow.keras import Input, optimizers, regularizers, callbacks
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import load_model, Model, Sequential
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from sklearn.metrics import mean_squared_error

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

import seaborn as sns
import pandas as pd
import numpy as np
import os

checkpoint_path = "models/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

def main():

    ## LOADING AND SCALING DATA ##
    dataframe = pd.read_csv("data/joined_data_2015-2017.csv", index_col=1)
    testdataframe = pd.read_csv("data/test_data_2018-2019.csv", index_col=1)

    dates = dataframe.index
    ref_forecasts = dataframe['DAF']

    variables = dataframe[["HH", "FH", "DD", "T", "AT"]]
    columns = list(variables)
    df = dataframe[columns].values.astype('float32')

    n_train_hours = 104 * 168

    df_train = df[:n_train_hours, :]
    df_test = df[n_train_hours:, :]

    scaler = MinMaxScaler().fit(df_train)
    train = scaler.transform(df_train)
    test = scaler.transform(df_test)

    ## TRAINING ##
    train_X, train_Y = train[:, :-1], train[:, -1]
    test_X, test_Y = test[:, :-1], test[:, -1]

    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)


    ## MODEL ##

    cp_callback = callbacks.ModelCheckpoint(filepath = checkpoint_path, save_weights_only=True, verbose=1)

    model = Sequential()
    model.add(LSTM(64, input_shape=(train_X.shape[1], train_X.shape[2],), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mae')
    model.summary()

    history = model.fit(train_X, train_Y, epochs=150, batch_size=24,
                        validation_data = (test_X, test_Y),
                        verbose=2, shuffle=False, callbacks=[cp_callback])


    forecast = model.predict(test_X)

    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

    inv_forecast = np.concatenate((test_X[:, -4:], forecast), axis=1)
    inv_forecast = scaler.inverse_transform(inv_forecast)
    inv_forecast = inv_forecast[:,4]

    test_Y = test_Y.reshape((len(test_Y), 1))
    inv_y = np.concatenate((test_X[:, -4:], test_Y), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,4]

    rmse = sqrt(mean_squared_error(inv_y, inv_forecast))
    reference_rmse = sqrt(mean_squared_error(inv_y, ref_forecasts[n_train_hours:]))
    print('Test RMSE: %.3f' % rmse)
    print('Reference RMSE: %.3f' % reference_rmse)

    dataframe_graph = pd.DataFrame({'date' : dates[n_train_hours:], 'real' : inv_y, 'forecast' : inv_forecast, 'ref_forecast' : ref_forecasts[n_train_hours:]})

    sns.lineplot(x= dataframe_graph['date'], y=dataframe_graph['real'], legend="full", label="Real")
    sns.lineplot(x= dataframe_graph['date'], y=dataframe_graph['forecast'], legend="full", label="Forecast")
    sns.lineplot(x= dataframe_graph['date'], y=dataframe_graph['ref_forecast'], legend="full", label="ref_Forecast")
    plt.show()


if __name__ == "__main__":
    main()
