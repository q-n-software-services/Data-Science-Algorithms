import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''# Initializations
weight = 72.12      # In Kilograms exact
height = 5.75       # In feet and decimals obtained by dividing inches by 12
body_type = 'Asian'     # Regional identification for the body type
exercise_time = 12.12   # In Minutes and decimal obtained from dividing seconds by 60
exercise_days_streak = 5    # Number of days for which the consistency is achieved in exercise/workout
food_intake = 500.12        # Number of Calories, intake of food
exhaustion_level = 0.12     # In Percentage i.e. out of 1
physical_condition = 'Male Teenager'    # Gender and age group
'''

# importing the Dataset
housing_data = pd.read_excel("Athelete Data Filtered.xlsx")
age_data = housing_data.iloc[:, 6].values
height_data = housing_data.iloc[:, 7].values
weight_data = housing_data.iloc[:, 8].values
pullups_data = housing_data.iloc[:, 20].values
length = len(age_data)
'''
print(length)
print("all data\n", housing_data)
print("age data", age_data)
print('height data', height_data)
print('weight data', weight_data)
print('pullups data', pullups_data)
'''

# Clustering Algorithm

# Initialisation

bmi = []
for i in range(length):
    this_bmi = (703 * weight_data[i])/(height_data[i] ** 2)
    bmi.append(this_bmi)


df = pd.DataFrame({
    'x': bmi,
    'y': pullups_data


})
colmap = {1: 'r', 2: 'g', 3: 'b', 4: 'y', 5: 'c'}

np.random.seed(200)

k = 5

# centroids[i] = [x,y]
centroids = {
    i+1: [np.random.randint(0, 120), np.random.randint(0, 220)]
    for i in range(k)
}


# Assignment Stage

def assignment(df, centroids):
    for i in centroids.keys():
        # sqrt((x1 - x2)^2 + (y1 - y2)^2)
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['x'] - centroids[i][0]) ** 2
                + (df['y'] - centroids[i][1]) ** 2
            )
        )

    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colmap[x])

    return df


df = assignment(df, centroids)
print(df)

# Update Stage

import copy

old_centroids = copy.deepcopy(centroids)


def update(k):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
    return k


centroids = update(centroids)

# Repeat Assignment Stage

df = assignment(df, centroids)

# Continue until all assigned categories don't change anymore

while True:
    closest_centroids = df['closest'].copy(deep=True)
    centroids = update(centroids)
    df = assignment(df, centroids)
    if closest_centroids.equals(df['closest']):
        break


fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])

plt.xlim(0, 120)
plt.ylim(0, 220)
plt.show()


my_weight = float(input("Enter you weight in pounds(lb)"))
my_height = float(input("Enter your height in inches"))
my_bmi = (703 * my_weight)/(my_height ** 2)
my_pullups = float(input("Enter the Number of Pullups That you perform on daily basis"))
cent_dist = 200
nearby_centroid = None
count = 1
for i in centroids.keys():
    dist = np.sqrt(((centroids[i][0] - my_bmi) ** 2) + ((centroids[i][1] - my_pullups) ** 2))
    print(centroids[i])
    if dist <= cent_dist:
        nearby_centroid = centroids[i]
        cent_dist = dist

centroid1 = [26.150985371252226, 16.999109263657957]
centroid2 = [25.799921783468605, 34.1019378816034]
centroid3 = [25.83033201500497, 50.167020148462356]
centroid4 = [26.03664882351105, 70.16129032258064]
centroid5 = [23.303039300426054, 174.33333333333334]
print("nearby centroid", nearby_centroid)
if nearby_centroid == centroid1:
    print("You have Poor Fitness")
    print("Either perform\t", centroid2[1], "\t Pull ups per day at an average")
    print("or increase your exercise time by\t", ((centroid2[1] - centroid1[1])/15)*10, "\t Minutes each day")
    print("Inorder to gain the level of Average Fitness")
elif nearby_centroid == centroid2:
    print("You have Average Fitness")
    print("Either perform\t", centroid3[1], "\t Pull ups per day at an average")
    print("or increase your exercise time by\t", ((centroid3[1] - centroid2[1]) / 15) * 10, "\t Minutes each day")
    print("Inorder to gain the level of Optimum Fitness")
elif nearby_centroid == centroid3:
    print("You have Optimum Fitness")
    print("Either perform\t", centroid4[1], "\t Pull ups per day at an average")
    print("or increase your exercise time by\t", ((centroid4[1] - centroid3[1]) / 15) * 10, "\t Minutes each day")
    print("Inorder to gain the level of Excellent Fitness")
elif nearby_centroid == centroid4:
    print("You have Excellent Fitness")
    print("Either perform\t", centroid5[1], "\t Pull ups per day at an average")
    print("or increase your exercise time by\t", ((centroid5[1] - centroid4[1]) / 15) * 10, "\t Minutes each day")
    print("Inorder to become a Fitness Freak")
elif nearby_centroid == centroid5:
    print("You are a Fitness Freak")
    print("Keep up your exercise routine")




