import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

dataset = [12, 22, 42, 15, 48, 42, 57, 12, 14, 13, 16, 19, 27, 19, 29, 26, 72, 112, 145, 15, 10, 1, -44]

outliers = []

quartile1, quartile3 = np.percentile(dataset, [25, 75])

print(quartile1, quartile3)

IQR_value = quartile3 - quartile1
print(IQR_value)

lower_bound_val = quartile1 - (IQR_value * 1.5)
upper_bound_val = quartile3 + (IQR_value * 1.5)

print(lower_bound_val, upper_bound_val)


