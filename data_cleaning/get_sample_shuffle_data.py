import csv
import os
import pandas as pd

path = "../weibo_senti_100k.csv"
savePath = "../shuffled_20000_weibo_senti.csv"
# set chunkSize and iterate loading the data preventing low memory problem

counter = 0

records = pd.read_csv(path, encoding='utf-8')
print("Loading")

shuffled_records = records.sample(n=20000)

shuffled_records.to_csv(savePath, encoding='utf-8',index=False)

print("Shuffle sample CSV File exported!")




