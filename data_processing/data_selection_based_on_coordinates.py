import csv
import pandas as pd
import numpy as np
import re
import os
import time
"""
data selection based on geographic coordinates

"""


def data_selection_based_on_coordinates(path, savePath, lat_min, lat_max, lon_min, lon_max):

    # set chunkSize and iterate loading the data preventing low memory problem
    chunkSize = 100000
    records = pd.read_csv(path, encoding='utf-8', chunksize=chunkSize)
    event = pd.DataFrame()

    for chunk in records:

        # ensure the datetime data type
        chunk['WGS84Latitude'] = pd.to_numeric(chunk['WGS84Latitude'])
        chunk['WGS84Longitude'] = pd.to_numeric(chunk['WGS84Longitude'])

        # for precedence reason of &, these must be () around
        # filter the geographic coordiantes based on the given location
        temp = chunk.loc[(chunk.WGS84Latitude > lat_min) & (chunk.WGS84Latitude < lat_max) & (chunk.WGS84Longitude> lon_min) & (chunk.WGS84Longitude<lon_max)]
        event = event.append(temp)

    # check if the file is already excite otherwise create one
    if not os.path.isfile(savePath):
        event.to_csv(savePath, encoding='utf-8',index=False)
        print("Create a new File")
    else:
        # if the file excites append the rest chunks on the file with no header
        #chunk.to_csv(savePath, encoding='utf-8', mode='a', header=False, index=False)
        print("The file is already excited")
        print("Append on the File")

    print("CSV File exported!")


if __name__ == '__main__':
    path = "../scored_cleaned_data_M_log.csv"
    savePath = "../Mercedes_Benz_Arena_whole_year.csv"

    # ---- Hongkou football stadium
    # lat_min = 31.2718
    # lat_max = 31.2743
    # lon_min = 121.4748
    # lon_max = 121.4777

    # ---Mercedes-Benz Arena
    lat_min = 31.1900
    lat_max = 31.1921
    lon_min = 121.4874
    lon_max = 121.4900

    data_selection_based_on_coordinates(path, savePath, lat_min, lat_max, lon_min, lon_max)