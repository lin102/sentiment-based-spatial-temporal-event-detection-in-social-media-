import csv
import pandas as pd
import numpy as np
import re
import os
import time
"""
data selection based on date

"""


def sentiment_data_selection_based_on_date_time(path, savePath, start: str, end: str):

    # set chunkSize and iterate loading the data preventing low memory problem
    chunkSize = 100000
    records = pd.read_csv(path, encoding='utf-8', chunksize=chunkSize)
    event = pd.DataFrame()

    for chunk in records:

        # ensure the datetime data type
        chunk['createdAT'] = pd.to_datetime(chunk['createdAT'])
        # for precedence reason of &, these must be () around
        temp = chunk.loc[(chunk.createdAT >= start) & (chunk.createdAT <= end)]
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
    #path = "../scored_cleaned_data_M_log.csv"
    path = '../scored_cleaned_data_M_log.csv'
    savePath = "../20140130.csv"
    start = '2014-01-30 00:00:00'
    end = '2014-01-31 00:00:00'
    sentiment_data_selection_based_on_date_time(path, savePath, start, end)