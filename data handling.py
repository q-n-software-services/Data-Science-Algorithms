import pandas as pd

my_data = pd.read_csv('C:\\Users\\HP\\Downloads\\archive\\flights.csv', low_memory=False)

print(my_data)
print()
print()
stats = my_data.describe()
print(stats)
stats.to_csv("stats.csv")
print()
print()

t_df = my_data.T
print(t_df)
print('\n\n\n')
df_to_np = my_data.to_numpy()
print(df_to_np)

print('\n\n\n')
print(df_to_np[0])

print('\n\n\n')
