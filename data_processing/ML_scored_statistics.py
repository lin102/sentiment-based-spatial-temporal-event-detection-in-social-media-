import os
import jieba
import numpy as np
import pandas as pd
import gensim
import joblib


path = "../scored_cleaned_data_M_log.csv"

# set chunkSize and iterate loading the data preventing low memory problem
chunkSize = 100000
chunk_counter = 0
records = pd.read_csv(path, encoding='utf-8', chunksize=chunkSize)
print("Loading")

positive =0
negative =0
none_value = 0

for chunk in records:
    # Iterate records

    chunk_counter = chunk_counter + 1
    print("This is the " + str(chunk_counter) + " chunk")

    positive = positive + chunk[chunk.M_log_score == 1].M_log_score.count()
    negative = negative + chunk[chunk.M_log_score == 0].M_log_score.count()
    none_value = none_value + chunk[chunk.M_log_score == -99].M_log_score.count()

print("positive: ", positive,"negative: ", negative, "null: ", none_value)
