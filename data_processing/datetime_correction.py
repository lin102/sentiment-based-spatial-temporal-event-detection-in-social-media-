import csv
import pandas as pd
import numpy as np
import re
import os
import time
import matplotlib.pyplot as plt

"""
there is 6 hours error in the datetime

"""

path = "../scored_cleaned_data_M_log.csv"
savePath = "../scored_cleaned_data_M_log_correct.csv"
# set chunkSize and iterate loading the data preventing low memory problem

records = pd.read_csv(path, encoding='utf-8')

difference = pd.Timedelta('8 hours')

records['createdAT'] = pd.to_datetime(records['createdAT'])
records['createdAT'] = records['createdAT'] + difference

records.to_csv(savePath, encoding='utf-8',index=False)