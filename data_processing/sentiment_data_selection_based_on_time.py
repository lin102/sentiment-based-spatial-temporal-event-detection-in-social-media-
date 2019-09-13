import csv
import pandas as pd
import numpy as np
import re
import os
import time
"""
data selection based on time

"""


def sentiment_data_selection_based_on_date_time(path, savePath, start: str, end: str, sentiment: int):

    # set chunkSize and iterate loading the data preventing low memory problem
    chunkSize = 100000
    records = pd.read_csv(path, encoding='utf-8', chunksize=chunkSize)
    event = pd.DataFrame()

    for chunk in records:

        # ensure the datetime data type
        chunk['createdAT'] = pd.to_datetime(chunk['createdAT'])
        # for precedence reason of &, these must be () around
        temp = chunk.loc[(chunk.createdAT >= start) & (chunk.createdAT <= end)&(chunk.M_log_score == sentiment)]
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
    path = '../Mercedes_Benz_Arena_whole_year.csv'
    savePath = "../Mercedes_Benz_Arena_20140511_negative.csv"
    start = '2014-05-11 00:00:00'
    end = '2014-05-12 00:00:00'
    sentiment = 0
    sentiment_data_selection_based_on_date_time(path, savePath, start, end, sentiment)