from tensorflow.keras import Input, optimizers, regularizers
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


def main():

    ## LOADING AND SCALING DATA ##
    dataframe = pd.read_csv("data/joined_data_2015-2017.csv", index_col=1)

    dates = dataframe.index
    ref_forecasts = dataframe['DAF']
    print(len(ref_forecasts))

    variables = dataframe[["AT", "FH", "DD", "T"]]
    columns = list(variables)
    df_train = dataframe[columns].values.astype('float32')

    scaler = MinMaxScaler(feature_range=(0,1))
    df_train_scaled = scaler.fit_transform(df_train)

    ## TRAINING ##
    n_train_hours = 104 * 168

    train = df_train_scaled[:n_train_hours, :]
    test = df_train_scaled[n_train_hours:, :]

    train_X, train_Y = train[:, :], train[:, 0]
    test_X, test_Y = test[:, :], test[:, 0]
    print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)


    ## MODEL ##
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(train_X.shape[1],
                                                      train_X.shape[2],),
                   return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(1))

    optimizer = optimizers.Adam(clipvalue=0.5, lr=1e-4)
    model.compile(optimizer=optimizer, loss='mse')
    model.summary()

    history = model.fit(train_X, train_Y, epochs=50, batch_size=16,
                        validation_data=(test_X,test_Y), verbose=2,
                        shuffle=False)


    forecast = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

    inv_forecast = np.concatenate((forecast, test_X[:, 1:]), axis=1)
    inv_forecast = scaler.inverse_transform(inv_forecast)

    inv_forecast = inv_forecast[:,0]

    test_Y = test_Y.reshape((len(test_Y), 1))
    inv_y = np.concatenate((test_Y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)

    inv_y = inv_y[:,0]

    rmse = sqrt(mean_squared_error(inv_y, inv_forecast))
    reference_rmse = sqrt(mean_squared_error(inv_y, ref_forecasts[:-n_train_hours]))
    print('Test RMSE: %.3f' % rmse)
    print('Reference RMSE: %.3f' % reference_rmse)

    dataframe_graph = pd.DataFrame({'date' : dates[:-n_train_hours], 'real' : inv_y, 'forecast' : inv_forecast, 'ref_forecast' : ref_forecasts[:-n_train_hours]})

    sns.lineplot(x= dataframe_graph['date'], y=dataframe_graph['real'], legend="full", label="Real")
    sns.lineplot(x= dataframe_graph['date'], y=dataframe_graph['forecast'], legend="full", label="Forecast")
    sns.lineplot(x= dataframe_graph['date'], y=dataframe_graph['ref_forecast'], legend="full", label="Ref. Forecast")
    plt.show()


if __name__ == "__main__":
    main()
