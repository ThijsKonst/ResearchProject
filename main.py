import seaborn as sns
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib
import time

import keras.applications
import keras.datasets
import keras.preprocessing
import keras.wrappers


from matplotlib import pyplot as plt
from keras.layers import Dense, LSTM, Input, Dropout, Bidirectional
from keras.models import Sequential
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras_self_attention import SeqSelfAttention
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from sklearn.metrics import mean_squared_error

matplotlib.use('TkAgg')

print("Num GPUs Available: ", tf.config.list_physical_devices())

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


def main(attention, bidi, show):

    start_time = time.time()
    # LOADING AND SCALING DATA ##
    dataframe = pd.read_csv("data/joined_data_2015-2019.csv",
                            parse_dates=['date'], index_col=1)

    dates = dataframe.index
    ref_forecasts = dataframe['DAF']
    dataframe = dataframe.reset_index()
    dataframe['WD'] = dataframe['date'].dt.dayofweek
    dataframe['MM'] = dataframe['date'].dt.month
    variables = dataframe[["HH", "WD", "FH", "DD", "T", "AT"]]
    columns = list(variables)
    df = dataframe[columns].values.astype('float32')

    n_train_hours = 26280

    df_train = df[:n_train_hours, :]
    df_test = df[n_train_hours:, :]

    scaler = MinMaxScaler().fit(df_train)
    train = scaler.transform(df_train)
    test = scaler.transform(df_test)

    # TRAINING
    train_X, train_Y = train[:-24, :], train[24:, -1]
    test_X, test_Y = test[:-24, :], test[24:, -1]

    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    # MODEL
    model = Sequential()
    if bidi:
        model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(train_X.shape[1], train_X.shape[2], )))
    else:
        model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2],),return_sequences=True))

    if attention:
        model.add(SeqSelfAttention(attention_activation='sigmoid'))

    model.add(Dropout(0.3))
    if bidi:
        model.add(Bidirectional(LSTM(64)))
    else:
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

    #es = EarlyStopping(monitor="val_loss", verbose=1, patience=20)

    callbacks = [LearningRateScheduler((lr_scheduler), verbose=1)]

    history = model.fit(train_X, train_Y, epochs=200, batch_size=24,
              validation_split=0.33, verbose=1, shuffle=False, callbacks=callbacks)

    if show:
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    # FORECASTING THE DATA DAY BY DAY

    n_days = int(len(test_X) / 24)

    forecast = np.array([])

    for i in range(n_days):
        test_batch = test_X[(i*24):((i+1)*24), :]
        prediction = model.predict(test_batch)
        forecast = np.append(forecast, prediction)

    forecast = forecast.reshape(forecast.shape[0], -1)

    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

    inv_forecast = np.concatenate((test_X[:, -5:], forecast), axis=1)
    inv_forecast = scaler.inverse_transform(inv_forecast)
    inv_forecast = inv_forecast[:, 5]

    test_Y = test_Y.reshape((len(test_Y), 1))
    inv_y = np.concatenate((test_X[:, -5:], test_Y), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 5]

    # SCORING METHODS

    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    rmse = sqrt(mean_squared_error(inv_y, inv_forecast))
    reference_rmse = sqrt(mean_squared_error(inv_y,
                                             ref_forecasts[n_train_hours+24:]))

    pcc = np.corrcoef(inv_y, inv_forecast)
    reference_pcc = np.corrcoef(inv_y, ref_forecasts[n_train_hours+24:])

    mape = mean_absolute_percentage_error(inv_y, inv_forecast)
    reference_mape = mean_absolute_percentage_error(inv_y, ref_forecasts[n_train_hours+24:])

    currentTime = str(int(time.time()))
    if attention and bidi:
        f = open("results/Test_results_BiLSTM_attention_" + currentTime, "x")
        pd.DataFrame(history.history).to_csv("results/Loss_BiLSTM_attention.csv")
    elif attention:
        f = open("results/Test_results_attention_" + currentTime, "x")
        pd.DataFrame(history.history).to_csv("results/Loss_attention.csv")
    elif bidi:
        f = open("results/Test_results_BiLSTM_" + currentTime, "x")
        pd.DataFrame(history.history).to_csv("results/Loss_BiLSTM.csv")
    else:
        f = open("results/Test_results_LSTM_" + currentTime, "x")
        pd.DataFrame(history.history).to_csv("results/Loss_LSTM.csv")


    print("written at " + currentTime)
    f.write('Test RMSE: %.3f \n' % rmse)
    f.write('Reference RMSE: %.3f \n' % reference_rmse)
    f.write('Test PCC: %.3f \n' % pcc[0, 1])
    f.write('Reference PCC: %.3f \n' % reference_pcc[0, 1])
    f.write('Test MAPE: %.3f \n' % mape)
    f.write('Reference MAPE: %.3f \n' % reference_mape)

    dataframe_graph = pd.DataFrame({'date': dates[n_train_hours+24:],
                                    'real': inv_y, 'forecast': inv_forecast,
                                    'ref_forecast':
                                    ref_forecasts[n_train_hours+24:]})

    if attention and bidi:
        dataframe_graph.to_csv('results/bilstm_attention/BiLSTM_prediction_'+ currentTime +'.csv')
    elif attention:
        dataframe_graph.to_csv('results/attention/Attention_prediction_' + currentTime +'.csv')
    elif bidi:
        dataframe_graph.to_csv('results/bilstm/BiLSTM_prediction_'+ currentTime + '.csv')
    else:
        dataframe_graph.to_csv('results/lstm/LSTM_prediction_'+ currentTime + '.csv')

    f.write(dataframe_graph.describe().to_string())
    f.close()
    if show:
        sns.lineplot(x=dataframe_graph['date'], y=dataframe_graph['real'],
                     legend="full", label="Real [MW]")
        sns.lineplot(x=dataframe_graph['date'], y=dataframe_graph['forecast'],
                     legend="full", label="Forecast [MW]")
        plt.show()

    return pcc, rmse, mape, (time.time()-start_time)

def Average(lst):
    return sum(lst)/len(lst)

def Test(modelName, attention, bidi, times):
    # Running the tests "times" times.
    with open("aggresults-test3-" + modelName, "x") as f:
        pccArray, rmseArray, mapeArray, timeArray = [],[],[],[]
        for i in range(times):
            result_pcc, result_rmse, result_mape, result_time = main(attention, bidi, False)
            pccArray.append(result_pcc[0,1])
            rmseArray.append(result_rmse)
            mapeArray.append(result_mape)
            timeArray.append(result_time)

        pcc = Average(pccArray)
        rmse = Average(rmseArray)
        mape = Average(mapeArray)
        exec_time = Average(timeArray)

        f.write(str(pccArray) + "\n")
        f.write(str(rmseArray)+ "\n")
        f.write(str(mapeArray)+ "\n")
        f.write(str(timeArray)+ "\n")

        f.write(modelName + " agg pcc: %.6f \n" % pcc)
        f.write(modelName + " agg RMSE: %.6f \n" % rmse)
        f.write(modelName + " agg MAPE: %.6f \n" % mape)
        f.write(modelName + " agg time: %.6f \n" % exec_time)
        f.write("\n")

if __name__ == "__main__":
    Test("AttentionBiLSTM", True, True, 10)
    Test("AttentionLSTM", True, False, 10)
    Test("BiLSTM", False, True, 10)
    Test("LSTM", False, False, 10)
