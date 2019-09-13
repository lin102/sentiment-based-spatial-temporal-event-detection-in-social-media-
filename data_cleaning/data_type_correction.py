import pandas as pd
import numpy as ny

path = "../cleaned_data.csv"
savePath = "../cleaned_data_setdatatype.csv"

records = pd.read_csv(path, encoding='utf-8')

records['createdAT'] = pd.to_datetime(records['createdAT'], errors='coerce',infer_datetime_format=True)

print(records.shape)
print(records.head())
print(records.dtypes)

records.to_csv(savePath,encoding='utf-8')