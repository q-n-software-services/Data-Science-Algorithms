import numpy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Initializations
weight = 72.12      # In Kilograms exact
height = 5.75       # In feet and decimals obtained by dividing inches by 12
body_type = 'Asian'     # Regional identification for the body type
exercise_time = 12.12   # In Minutes and decimal obtained from dividing seconds by 60
exercise_days_streak = 5    # Number of days for which the consistency is achieved in exercise/workout
food_intake = 500.12        # Number of Calories, intake of food
exhaustion_level = 0.12     # In Percentage i.e. out of 1
physical_condition = 'Male Teenager'    # Gender and age group


# importing the Dataset
housing_data = pd.read_excel("E:\\Machine Learning\\Athelete Data Filtered.xlsx")
age_data = housing_data.iloc[:, 6].values
height_data = housing_data.iloc[:, 7].values
weight_data = housing_data.iloc[:, 8].values
pullups_data = housing_data.iloc[:, 20].values
length = len(age_data)

print(length)
print("all data\n", housing_data)
print("age data", age_data)
print('height data', height_data)
print('weight data', weight_data)
print('pullups data', pullups_data)
bmi = []
for i in range(length):
    this_bmi = (703 * weight_data[i])/(height_data[i] ** 2)
    bmi.append(this_bmi)

# Clustering Algorithm

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'x': bmi,
    'y': pullups_data


})
colmap = {1: 'r', 2: 'g', 3: 'b', 4: 'y', 5: 'c'}

kmeans = KMeans(n_clusters=5)
kmeans.fit(df)

labels = kmeans.predict(df)
centroids = kmeans.cluster_centers_

fig = plt.figure(figsize=(5, 5))
colors = map(lambda x: colmap[x + 1], labels)
colors1 = list(colors)
plt.scatter(df['x'], df['y'], color=colors1, alpha=0.5, edgecolor='k')
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid, color=colmap[idx + 1])
plt.xlim(0, 120)
plt.ylim(0, 220)
plt.show()

pred = kmeans.predict([[27, 32]])
print(pred)
