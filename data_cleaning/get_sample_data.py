import csv
import os
"""
Extract a small part of original file to see the data structure
for further data cleaning

"""
path = "../weibo_senti_100k.csv"
savePath = "../Sample1000_weibo_senti_100k.csv"

i = 0
# the counter

with open(path, newline='', encoding='utf-8') as csvFile:
    csvReader = csv.reader(csvFile)
    #iteration
    for row in csvReader:
        print(row)
        with open(savePath, 'a', newline='', encoding='utf-8') as sampleFile:
            writer = csv.writer(sampleFile)
            writer.writerow(row)
            i = i + 1
            # get first 1000 rows as sample data
            if i == 1000:
                print("Sample data got!")
                break




