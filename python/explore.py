import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# print(os.getcwd() + "\n")

# load data
train = pd.read_csv("../data/train.csv")

# explore data
train.describe()
# train.loc[1:10, 'Age']
# train[train.Age > 50].loc[:,'Age']
# train[train.Age > 50]['Age']

train.groupby('Pclass').mean()
train.groupby('Pclass').size()

# clean data
meanAge = train.Age.mean()
train.Age = train.Age.fillna(meanAge)


# visualization
plt.scatter(y=train.Fare, x=train.Age, alpha=0.2, c=train.Pclass)

plt.minorticks_on()
plt.grid(b=True, which='major', c='gray', linestyle='-')
plt.grid(b=True, which='minor', c='lightgray', linestyle='--')
plt.xlim([-5,85])
plt.xlabel('Age')