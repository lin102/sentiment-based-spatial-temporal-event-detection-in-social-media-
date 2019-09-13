import csv
import os
import pandas as pd

path = "cleaned_data.csv"
savePath = "add index.csv"

records = pd.read_csv(path, encoding='utf-8')
print("Shape of original records", records.shape)
print("len: ", len(records.index))

index = list(range(1, len(records.index)+1))

# insert the index column to the left
records.insert(0, "recordID",index)

records.to_csv(savePath, encoding='utf-8', index=False)
