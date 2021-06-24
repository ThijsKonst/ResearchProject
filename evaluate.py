import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')

attention = pd.read_csv('results/Attention_prediction.csv', parse_dates=['date'])
bilstm = pd.read_csv('results/BiLSTM_prediction.csv')
lstm = pd.read_csv('results/LSTM_prediction.csv')
bilstmAttention = pd.read_csv('results/BiLSTM_Attention_prediction.csv')
dataframe_graph = pd.DataFrame({'Time (h)': attention['date'], 'attention': attention['forecast'], 'bilstm' : bilstm['forecast'], 'lstm' : lstm['forecast'], 'bilstm_attention': bilstmAttention['forecast'], 'reference' : lstm['ref_forecast'], 'Energy consumption [MW]': lstm['real']})

sns.lineplot(x=dataframe_graph['Time (h)'], y=dataframe_graph['Energy consumption [MW]'],
             legend="full", label="Real [MW]")
sns.lineplot(x=dataframe_graph['Time (h)'], y=dataframe_graph['attention'],
             legend="full", label="Attention [MW]")
sns.lineplot(x=dataframe_graph['Time (h)'], y=dataframe_graph['bilstm'],
             legend="full", label="BLSTM [MW]")
sns.lineplot(x=dataframe_graph['Time (h)'], y=dataframe_graph['lstm'],
             legend="full", label="LSTM [MW]")
sns.lineplot(x=dataframe_graph['Time (h)'], y=dataframe_graph['bilstm_attention'],
             legend="full", label="BLSTM Attention [MW]")

sns.lineplot(x=dataframe_graph['Time (h)'], y=dataframe_graph['reference'],
             legend="full", label="TSO baseline [MW]")
#plt.show()

plt.clf()

attention = pd.read_csv('results/Loss_attention.csv')
bilstm = pd.read_csv('results/Loss_BiLSTM.csv')
lstm = pd.read_csv('results/Loss_LSTM.csv')
bilstmAttention = pd.read_csv('results/Loss_BiLSTM_attention.csv')

def makeDF(dataset, name):
    dataframe_loss = pd.DataFrame({'Epoch (#)': dataset['epoch'],
                                'loss': dataset['loss'],
                                'class': name + "_loss"})

    dataframe_val_loss = pd.DataFrame({'Epoch (#)': dataset['epoch'],
                                   'loss': dataset['val_loss'],
                                   'class': name + "_val_loss"})

    dataframe_loss = pd.concat([dataframe_loss, dataframe_val_loss])

    return dataframe_loss

dataframe_loss_attention = makeDF(attention, 'attention')
dataframe_loss_bilstm = makeDF(bilstm, 'bilstm')
dataframe_loss_lstm = makeDF(lstm, 'lstm')
dataframe_loss_bilstm_attention = makeDF(bilstmAttention, 'bilstm_attention')


sns.lineplot(data=dataframe_loss_attention, x='Epoch (#)', y='loss', style='class', label="attention", legend='brief')
sns.lineplot(data=dataframe_loss_bilstm, x='Epoch (#)', y='loss', style='class', label="bilstm", legend='brief')
sns.lineplot(data=dataframe_loss_lstm, x='Epoch (#)', y='loss', style='class', label="lstm", legend="brief")
sns.lineplot(data=dataframe_loss_bilstm_attention, x='Epoch (#)', y='loss', style='class', label="bilstm_attention", legend='brief')

#ax2 = plt.twinx()
#sns.lineplot(data=dataframe_loss, x='Epoch (#)', y='learning rate', label="learning rate", color='y', ax=ax2)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()

