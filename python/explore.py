import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

# use prettier ggplot style
matplotlib.style.use('ggplot')

# print(os.getcwd() + "\n")

# load data
train = pd.read_csv("../data/train.csv")

# EXPLORE DATA
train.describe(include='all')

# subsetting
# train.loc[1:10, 'Age']
# train[train.Age > 50].loc[:,'Age']
# train[train.Age > 50]['Age']

# print counts per category
for col in ['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']:
    print train[col].value_counts(dropna=False) # sorts by counts descending
#    print train.groupby(col).size() # sorts by categories ascending

# descriptive statistics
train.groupby('Pclass').mean()

# clean data (fill missing Embark and Age values)
train.Embarked.fillna('S', inplace=True) # assume 'S'
meanAge = train.Age.mean()
train.Age = train.Age.fillna(meanAge)

# convert to categories (requires pandas 0.15+)
for col in ['Survived', 'Pclass', 'Sex', 'Embarked']:
    train[col] = train[col].astype('category')

# Feature Correlation

corr = train.corr()
sns.heatmap(corr, square=True, annot=True, linewidths='1', cmap="RdBu")
plt.savefig('feature_correlation.png')

# visualization with matplotlib
plt.scatter(y=train.Fare, x=train.Age, alpha=0.2, c=train.Pclass)

plt.minorticks_on()
plt.grid(b=True, which='major', c='gray', linestyle='-')
plt.grid(b=True, which='minor', c='lightgray', linestyle='--')
plt.xlim([-5,85])
plt.xlabel('Age')
plt.show()


# Random Forest

# must encode categories as integers
sexEncoder = preprocessing.LabelEncoder()
train['Sex'] = sexEncoder.fit_transform(train.Sex)
embEncoder = preprocessing.LabelEncoder()
train['Embarked'] = embEncoder.fit_transform(train.Embarked)

columns = np.array(['Sex', 'Pclass', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked'])

forest = RandomForestClassifier(n_estimators=1000, max_depth=5)
fit = forest.fit(train[columns], train.Survived)

# analyze variable importance
# (inspired by http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html)
varImportance = pd.Series(fit.feature_importances_, index=columns).sort_values()
print varImportance

varImportance.plot.barh() # color='skyblue'
ax = plt.gca()
ax.yaxis.grid(False)
#ax.set_axisbelow(True)
#ax.set_axis_bgcolor('darkgrey')
#ax.xaxis.grid(color='white', linestyle='solid', linewidth='1')
#ax.set_frame_on(False)
plt.tick_params(axis='y', left='off', right='off')
plt.tick_params(axis='x', top='off', direction='out')
plt.title('Variable Importance')
plt.xlim(-0.015, 0.515)
plt.show()

plt.savefig('variable_importance.png')
