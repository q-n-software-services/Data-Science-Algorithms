import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

from sklearn.datasets import fetch_california_housing
ca = fetch_california_housing()

dir(ca)
X = ca.data
y = ca.target

data = np.concatenate((X, y.reshape(-1, 1)), axis=1)

df = pd.DataFrame(data, columns=ca.feature_names+['target'])
print(df)

df.drop(['Latitude', 'Longitude'], axis=1, inplace=True)

features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup',]
print(features)


sns.set_style('dark')
for col in features:
    plt.figure(figsize=(15, 4))
    plt.subplot(131)
    sns.distplot(df[col], label='skew: ' + str(np.round(df[col].skew(), 2)))
    plt.legend()
    plt.subplot(132)
    sns.boxplot(df[col])
    plt.subplot(133)
    stats.probplot(df[col], plot=plt)
    plt.tight_layout()

    # Un-Comment the following line of code to view the Graphs/Plots of data

    # plt.show()


# Capping using Percentile Method
# Winsorization technique

df_cap = df.copy()


def percentile_capping(df, cols, from_low_end, from_high_end):
    for col in cols:
        # The following code is replaced by one line of Scipy function
        '''lower_bound = df[col].quantile(from_low_end)
        upper_bound = df[col].quantile(1 - from_high_end)

        df[col] = np.where(df[col] > upper_bound, upper_bound, np.where(df[col] < lower_bound, lower_bound, df[col]))'''

        stats.mstats.winsorize(a=df[col], limits=(from_low_end, from_high_end), inplace=True)


percentile_capping(df_cap, features, 0.01, 0.01)
print(df_cap.describe())


for col in features:
    plt.figure(figsize=(16, 4))

    plt.subplot(141)
    sns.distplot(df[col], label='skew: ' + str(np.round(df[col].skew(), 2)))
    plt.legend()

    plt.subplot(142)
    sns.distplot(df_cap[col], label='skew: ' + str(np.round(df[col].skew(), 2)))
    plt.legend()

    plt.subplot(143)
    sns.boxplot(df[col])
    plt.title("Before")

    plt.subplot(133)
    stats.probplot(df[col], plot=plt)
    plt.title("After")
    plt.tight_layout()

    # Un-Comment the following line of code to view the Graphs/Plots of data

    # plt.show()










