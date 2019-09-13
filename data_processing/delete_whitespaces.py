import csv
import pandas as pd
import numpy as np
import re
import os
import time
"""
Delete whitespaces

"""

path = "../cleaned_chinese_dropNull.csv"
savePath = "../cleaned_droppedWhitespaces.csv"
# set chunkSize and iterate loading the data preventing low memory problem
chunkSize = 100000
counter = 0
timer = 0

records = pd.read_csv(path, encoding='utf-8', chunksize=chunkSize)
print("Loading")

for chunk in records:
    # Iterate records
    counter = counter + 1
    print("This is the " + str(counter) + " chunk")

    # store all the empty msgText row index for deleting later
    # using iterator to Iterate rows
    for index, row in chunk.iterrows():
        msgText = chunk.loc[index, 'msgtext']
        # delete all the whitespaces and join all the words together
        msgText = "".join(msgText.split())
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
