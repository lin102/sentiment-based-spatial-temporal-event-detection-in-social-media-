import csv
import pandas as pd
import numpy as np
import re
import os
import time
import matplotlib.pyplot as plt

"""
population sentiment time series analysis

"""


def sentiment_time_series_analysis(path):
    records = pd.read_csv(path, encoding='utf-8')
    records = records[['createdAT','M_log_score']]

    # ensure the datetime data type
    records['createdAT'] = pd.to_datetime(records['createdAT'])
    records = records.set_index('createdAT')

    positive_number = records[records['M_log_score'] == 1].M_log_score.resample('D').count()
    negative_number = records[records['M_log_score'] == 0].M_log_score.resample('D').count()
    # print(positive_number) # series type

    timing_analysis: pd.DataFrame = pd.concat([positive_number, negative_number], axis=1)
    timing_analysis.columns = ['positive_number','negative_number']
    timing_analysis['sentiment_ratio'] = timing_analysis.positive_number / timing_analysis.negative_number

    # for small scale the division is not ok anymore because the sample number maybe too small
    #timing_analysis['sentiment_ratio'] = timing_analysis.positive_number - timing_analysis.negative_number
    # print(timing_analysis)

    # filter all the low sampled data
    timing_analysis = timing_analysis[timing_analysis['positive_number'] + timing_analysis['negative_number'] >40]


    # ------- Plot the time series ------------
    plt.plot(timing_analysis.index, timing_analysis.sentiment_ratio,
             marker ='o', color = 'blue', markersize = 5,)
    plt.xlabel('Date Time')
    plt.ylabel('Sentiment Orientation')
    plt.title("Weibo Sentiment Tendency")
    plt.show()


if __name__ == '__main__':

    #path = "../scored_cleaned_data_M_log.csv"
    path = "../Mercedes_Benz_Arena_whole_year.csv"
    sentiment_time_series_analysis(path)
