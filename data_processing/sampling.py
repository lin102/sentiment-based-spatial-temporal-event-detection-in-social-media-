import pandas as pd

path = "../scored_cleaned_data_M_log.csv"
sample1SavePath = ""
#sample2SavePath = "../sample2000b.csv"

records = pd.read_csv(path, encoding='utf-8')
print("sampling")
#sample1 = records.sample(2000)
#sample2 = records.sample(2000)
#sample1.to_csv(sample1SavePath, encoding='utf-8', index=False)
#sample2.to_csv(sample2SavePath, encoding='utf-8',index=False)
print(records.tail(100))

print("1000 sample CSV File exported!")
