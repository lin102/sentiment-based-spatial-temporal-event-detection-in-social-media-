import csv
import pandas as pd
import numpy as np
import re
import os
import time
from collections import defaultdict
import jieba

"""
data selection based on the key words

"""


def data_selection_based_on_key_words(path:str, save_path:str, key_words_list:list) ->pd.DataFrame():

    records = pd.read_csv(path, encoding='utf-8')
    key_word_records = pd.DataFrame()

    # iterate the rows to filter the rows containing the key words
    for index, row in records.iterrows():

        seg_list = jieba.cut(row['msgtext'])
        for x in seg_list:
            if x in key_words_list:
                key_word_records = key_word_records.append(row)
    # save filtered records
    key_word_records.to_csv(save_path, encoding='utf-8', index=False)
    print("File Exported!")
    return key_word_records


if __name__ == '__main__':

    path = '../high_peak_negative_2014.csv'
    save_path = '../transport_2014.csv'
    key_words_list = ['机场','地铁站','飞机','登机','晚点','地铁','轨道交通','火车','火车站']
    data_selection_based_on_key_words(path,save_path,key_words_list)