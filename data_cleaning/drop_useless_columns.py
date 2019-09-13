import pandas as pd
import numpy as ny

path = "../weibo_senti_100k.csv"
savePath = "../weibo_senti_100k_new.csv"

records = pd.read_csv(path, encoding='utf-8')
print("Shape of original records", records.shape)
print("columns", records.columns)
print("describe", records.describe())
print("Heads of records", records.head())
print("Tails of records", records.tail())

drop_column = ['Unnamed: 0']
#drop_column = ['Unnamed: 0.1','Unnamed: 0.1.1','Unnamed: 0.1.1','msgmid','userprovince','usercity','usercreated_at']
records = records.drop(drop_column, axis=1)
print("after drop",  records.columns)
records.to_csv(savePath, encoding='utf-8', index=False)