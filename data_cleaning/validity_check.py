import pandas as pd
import numpy as ny

path = "../after_dropColumns.csv"
savePath = "../cleaned_data.csv"

records = pd.read_csv(path, encoding='utf-8')

#records['WGS84Latitude'] = pd.to_numeric(records['WGS84Latitude'], errors='ignore')
#records['WGS84Longitude'] = pd.to_numeric(records['WGS84Longitude'], errors='ignore')
# set the data type of coordinates to numeric
records[["WGS84Latitude", "WGS84Longitude"]] = records[["WGS84Latitude", "WGS84Longitude"]].apply(pd.to_numeric, errors='coerce')

print(records['WGS84Longitude'].dtype)

# Geographic coordinate validity check
afterFilterDF = records[(records["WGS84Latitude"]<34) & (records["WGS84Latitude"]>29) & (records["WGS84Longitude"]<124) & (records["WGS84Longitude"]>118)]

drop_column = ['idNearByTimeLine','distance']
afterFilterDF = afterFilterDF.drop(drop_column, axis=1)

print("final ", afterFilterDF.shape)
# to csv File
afterFilterDF.to_csv(savePath, encoding='utf-8')