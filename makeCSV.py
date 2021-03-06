from pandas import read_csv
from datetime import datetime, timedelta


def parser(x, y):
    if y == "24":
        y = "0"
        return datetime.strptime(x + y, '%Y%m%d%H') + timedelta(days=1)
    return datetime.strptime(x + y, '%Y%m%d%H')


energyframe = read_csv('data/2018-2019.csv')
# Removing daylight saving hours from dataset
energyframe = energyframe.dropna()
energyframe.reset_index(inplace=True, drop=True)

# Group 4 times 15 minutes into an hour
energyframe = energyframe.groupby(energyframe.index // 4).sum()
print(energyframe)

priceframe = read_csv('data/prices2018-2019.csv')
priceframe = priceframe.drop("BZN|NL", axis=1)
priceframe.reset_index(inplace=True, drop=True)
priceframe = priceframe.fillna(priceframe.mean(axis=0))

print(priceframe)

dataset = read_csv("data/uurgeg_260_2011-2020.csv",
                   parse_dates={"date": ['YYYYMMDD', 'HH']},
                   keep_date_col=True,
                   date_parser=parser)

dataset = dataset.drop("STN", axis=1)

dataset.index.name = 'date'
# Only selecting the needed years of data
mask = (dataset['date'] > '2019-01-01') & (dataset['date'] <
                                           '2020-01-01 01:00:00')

dataset = dataset[mask]
dataset.reset_index(inplace=True, drop=True)

print(dataset)

# Merging the two datasets :)
dataset = dataset.merge(energyframe[["AT","DAF"]], left_index=True, right_index=True)
dataset = dataset.merge(priceframe[["Day-ahead Price [EUR/MWh]"]], left_index=True, right_index=True)


dataset.to_csv("data/validData.csv")
