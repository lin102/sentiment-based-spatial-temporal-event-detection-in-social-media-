import pandas as pd
import numpy as ny

path = "../8w_labeled_data/weibo_senti_100k.csv"

# skip error lines
records = pd.read_csv(path, encoding='utf-8', error_bad_lines= False)
print("Shape of original records", records.shape)
print("columns", records.columns)
print("describe", records.describe())
print("Heads of records", records.head())
print("Tails of records", records.tail())

# blink field give NA
# export to after_clean.csv
records.to_csv("../weibo_senti_100k.csv",na_rep="NA", encoding='utf-8')

#records_afterSelect = records[records['msgtext'].str.contains("饺子")]
#print("Shape after selection", records_afterSelect)
#records_afterSelect.to_csv(savePath, encoding='utf-8')

#records_afterSelect = records[records['msgtext'].str.contains("火锅")]
#print("Shape after selection", records_afterSelect)
#records_afterSelect.to_csv(savePath, encoding='utf-8')


