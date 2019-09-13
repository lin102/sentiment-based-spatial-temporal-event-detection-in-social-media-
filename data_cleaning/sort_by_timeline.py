import pandas as pd
import numpy as ny

path = "../cleaned_data.csv"
savePath = "../cleaned_data_after_sort.csv"

records = pd.read_csv(path, encoding='utf-8')


print(records['createdAT'].dtype)
print("Before sort", records['createdAT'].describe())
# Sort data by createdAT date timeline

# Sort dataFrame by the "createdAT" date timeline values
records = records.sort_values(by=['createdAT'])

print("After sort", records['createdAT'].describe())

records.to_csv(savePath,encoding='utf-8')