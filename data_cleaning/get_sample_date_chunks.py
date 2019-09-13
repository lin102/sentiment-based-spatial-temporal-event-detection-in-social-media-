import csv
import os
import pandas as pd

path = "../scored_cleaned_data.csv"
savePath = "../sample1000_scored_data_dic.csv"
# set chunkSize and iterate loading the data preventing low memory problem
chunkSize = 1000
counter = 0

records = pd.read_csv(path, encoding='utf-8', chunksize=chunkSize)
print("Loading")


for chunk in records:
    # Iterate records
    '''
    for i in range(len(records)):
        # parse the text to string type in case NULL
        msgText = str(records.loc[i, 'msgtext'])
        # using regular expression delete all after http
        msgText = re.sub(r'https?://.+', "", msgText)
        print(msgText)
        records.loc[i, 'msgtext'] = msgText
    '''
    counter = counter + 1
    print("This is the " + str(counter) + " chunk")
    chunk.to_csv(savePath, encoding='utf-8',index=False)
    break

print("1000 sample CSV File exported!")
