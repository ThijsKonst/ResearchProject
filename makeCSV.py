from pandas import read_csv
from datetime import datetime, timedelta

def parser(x, y):
    if y == "24":
        y = "0"
        return datetime.strptime(x + y, '%Y%m%d%H') + timedelta(days=1)
    return datetime.strptime(x + y, '%Y%m%d%H')

energyframe = read_csv('data/2015-2016.csv').append(read_csv('data/2016-2017.csv'))
# Removing daylight saving hours from dataset
energyframe = energyframe.dropna()
energyframe.reset_index(inplace = True, drop=True)

# Group 4 times 15 minutes into an hour
energyframe = energyframe.groupby(energyframe.index // 4).sum()
print(energyframe)


dataset = read_csv("data/uurgeg_260_2011-2020.csv",
                   parse_dates={"date" : ['YYYYMMDD','HH']},
                   date_parser=parser)

dataset = dataset.drop("STN", axis=1)

dataset.index.name ='date'
# Only selecting the needed years of data
mask = (dataset['date'] > '2015-01-01') & (dataset['date'] <
                                           '2017-01-01 01:00:00')

dataset = dataset[mask]
dataset.reset_index(inplace = True, drop=True)

print(dataset)

# Merging the two datasets :)
dataset = dataset.merge(energyframe[["AT","DAF"]], left_index=True, right_index=True)

dataset.to_csv("data/joined_data_2015-2017.csv")
