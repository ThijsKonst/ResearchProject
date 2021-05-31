import seaborn as sns
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib
import time
import ModelTrunk

from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from sklearn.metrics import mean_squared_error

matplotlib.use('TkAgg')

attention = False
print("Num GPUs Available: ", tf.config.list_physical_devices())

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


def main():
    # LOADING AND SCALING DATA ##
    dataframe = pd.read_csv("data/joined_data_2015-2019.csv",
                            parse_dates=['date'], index_col=1)

    dates = dataframe.index
    ref_forecasts = dataframe['DAF']
    dataframe = dataframe.reset_index()
    dataframe['WD'] = dataframe['date'].dt.dayofweek
    dataframe['MM'] = dataframe['date'].dt.month
    variables = dataframe[["HH", "WD", "MM", "FH", "DD", "T", "AT"]]
    columns = list(variables)
    df = dataframe[columns].values.astype('float32')
    df = df[8784:, :]
    ref_forecasts = ref_forecasts[8784:]
    dates = dates[8784:]

    n_train_hours = 26280 - 8784

    df_train = df[:n_train_hours, :]
    df_test = df[n_train_hours:, :]

    scaler = MinMaxScaler().fit(df_train)
    train = scaler.transform(df_train)
    test = scaler.transform(df_test)

    # TRAINING
    train_X, train_Y = train[:, :-1], train[:, -1]
    test_X, test_Y = test[:, :-1], test[:, -1]

    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)

    # MODEL

    if attention:
        inp = Input(shape=(train_X.shape[1], train_X.shape[2]))
        x = ModelTrunk.ModelTrunk(num_layers=4, dropout=0.1)(inp)
        x = Dense(1)(x)
        x = Dense(1)(x)

        model = Model(inp, x)
    else:
        model = Sequential()
        model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2],),
                  return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(64))
        model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    model.summary()

    def lr_scheduler(epoch, lr, warmup_epochs=15, decay_epochs=150, initial_lr=1e-6, base_lr=1e-3, min_lr=5e-5):
        if epoch <= warmup_epochs:
            pct = epoch / warmup_epochs
            return ((base_lr - initial_lr) * pct) + initial_lr

        if epoch > warmup_epochs and epoch < warmup_epochs+decay_epochs:
            pct = 1 - ((epoch - warmup_epochs) / decay_epochs)
            return ((base_lr - min_lr) * pct) + min_lr
        return min_lr

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    if attention:
        callbacks = [LearningRateScheduler((lr_scheduler), verbose=1)]
    else:
        callbacks = [es]

    history = model.fit(train_X, train_Y, epochs=200, batch_size=24,
              validation_data=(test_X, test_Y), verbose=1, shuffle=False, callbacks=callbacks)

    if attention:
        model.save_weights("models/attention.h5")
    else:
        model.save("models/lstm.h5")

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    forecast = model.predict(test_X)

    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

    print(forecast.shape)
    inv_forecast = np.concatenate((test_X[:, -6:], forecast), axis=1)
    print(inv_forecast.shape)
    inv_forecast = scaler.inverse_transform(inv_forecast)
    inv_forecast = inv_forecast[:, 6]

    test_Y = test_Y.reshape((len(test_Y), 1))
    inv_y = np.concatenate((test_X[:, -6:], test_Y), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 6]

    rmse = sqrt(mean_squared_error(inv_y, inv_forecast))
    reference_rmse = sqrt(mean_squared_error(inv_y,
                                             ref_forecasts[n_train_hours:]))

    currentTime = str(int(time.time()))
    if attention:
        f = open("results/Test_results_attention_" + currentTime,
                 "x")
    else:
        f = open("results/Test_results_LSTM_" + currentTime, "x")

    print("written at " + currentTime)
    f.write('Test RMSE: %.3f \n' % rmse)
    f.write('Reference RMSE: %.3f \n' % reference_rmse)

    dataframe_graph = pd.DataFrame({'date': dates[n_train_hours:],
                                    'real': inv_y, 'forecast': inv_forecast,
                                    'ref_forecast':
                                    ref_forecasts[n_train_hours:]})

    f.write(dataframe_graph.describe().to_string())
    f.close()
    sns.lineplot(x=dataframe_graph['date'], y=dataframe_graph['real'],
                 legend="full", label="Real [MW]")
    sns.lineplot(x=dataframe_graph['date'], y=dataframe_graph['forecast'],
                 legend="full", label="Forecast [MW]")
    #sns.lineplot(x=dataframe_graph['date'], y=dataframe_graph['ref_forecast'],
    #             legend="full", label="ref_Forecast [MW]")
    plt.show()


if __name__ == "__main__":
    main()
