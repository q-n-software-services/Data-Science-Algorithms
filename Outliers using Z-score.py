import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

dataset = [12, 22, 42, 15, 48, 42, 57, 12, 14, 13, 16, 19, 27, 19, 29, 26, 72, 112, 145, 15, 10, 1, -44]

outliers = []


def detect_outliers(data):
    threshold = 3
    mean = np.mean(data)
    std = np.std(data)

    for i in data:
        z_score = (i - mean) / std
        if np.abs(z_score) > threshold:
            outliers.append(i)

    return outliers


outlier_pt = detect_outliers(dataset)
print(outlier_pt)



