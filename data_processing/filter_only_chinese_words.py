import csv
import pandas as pd
import numpy as np
import re
import os
import time
"""
Using regular expression delete URL

"""

path = "../8w_labeled_data/weibo_senti_100k.csv"
savePath = "weibo_senti_100k.csv"
# set chunkSize and iterate loading the data preventing low memory problem
chunkSize = 100000
counter = 0
timer = 0

records = pd.read_csv(path, encoding='utf-8', chunksize=chunkSize)
print("Loading")

emptyMsg =[]
for chunk in records:

    # Iterate records
    counter = counter + 1
    print("This is the " + str(counter) + " chunk")

    # store all the empty msgText row index for deleting later
    # using iterator to Iterate rows
    for index, row in chunk.iterrows():
        msgText = chunk.loc[index, 'msgtext']
        # in some case the msgText is NULL then the type is float
        if type(msgText) is not str:
            msgText = str(msgText)
            #emptyMsg.append(index)
            #print("The " + str(index) + "MsgText is a empty one")
        # Only left Chinese character
        msgText = re.sub(r'[^\u4e00-\u9fa5]', '', msgText)
        chunk.loc[index, 'msgtext'] = msgText
        timer = timer + 1
        if timer % 4000 == 0:
            print(timer)

    # check if the file is already excite otherwise create one
    if not os.path.isfile(savePath):
        chunk.to_csv(savePath, encoding='utf-8',index=False)
        print("Create a new File")
    else:
        # if the file excites append the rest chunks on the file with no header
        chunk.to_csv(savePath, encoding='utf-8', mode='a', header=False, index=False)
        print("Append on the File")

print("Iteration finished")
print("CSV File exported!")
