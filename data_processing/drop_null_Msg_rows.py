import csv
import pandas as pd
import numpy as np
import re
import os
import time
"""
delete rows that msgtext column is NULL

"""

path = "../cleaned_only_chinese.csv"
savePath = "../cleaned_chinese_dropNull.csv"
# set chunkSize and iterate loading the data preventing low memory problem
chunkSize = 100000
chunk_counter = 0
row_counter = 0

records = pd.read_csv(path, encoding='utf-8', chunksize=chunkSize)
print("Loading")

for chunk in records:

    # Iterate records
    chunk_counter = chunk_counter + 1
    print("This is the " + str(chunk_counter) + " chunk")

    # store all the empty msgText row index for deleting later
    # using iterator to Iterate rows
    for index, row in chunk.iterrows():

        msgText = chunk.loc[index, 'msgtext']
        # strop delete the start and tail white spaces
        if msgText.strip() == "":
            print("This is a NULL", index)
            # inplace = true means modify the Original dataframe.
            chunk.drop(index, inplace=True)
        row_counter = row_counter + 1
        if row_counter % 4000 == 0:
            print(row_counter)

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
