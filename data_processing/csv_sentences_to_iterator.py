import jieba
import pandas as pd
import time
import jieba.posseg as pseg
import os
import profile
import gensim
from gensim.models import word2vec
import nltk
import cython

"""
This python MySentences class is creating a generator(iterator)
for word2Vec to genreate the word vector
"""


class CSVSentencesToIterator:

    # constructor function contains all the member variables, user need to input the csv file path
    def __init__(self, savePath):
        self.__savePath = savePath
        # set the chunk size for iteratively load the huge csv file
        self.__chunk_size = 100000
        self.__chunk_counter = 0
        self.__row_counter = 0
        self.__records = pd.read_csv(self.__savePath, encoding='utf-8', chunksize=self.__chunk_size)
        print("Loading")

    def __iter__(self):

        for chunk in self.__records:
            # Iterate records

            self.__chunk_counter = self.__chunk_counter + 1
            print("This is the " + str(self.__chunk_counter) + " chunk")

            # store all the empty msgText row index for deleting later
            # using iterator to Iterate rows
            for index, row in chunk.iterrows():
                msgText = chunk.loc[index, 'msgtext']
                # in some case the msgText is NULL then the type is float

                # sentences segmentation
                seg_list = jieba.cut(msgText)
                seg_list = list(seg_list)
                #print(seg_list)
                # yield is a key word for a generator in python
                yield seg_list

                self.__row_counter = self.__row_counter + 1
                if self.__row_counter % 5000 == 0:
                    print(self.__row_counter)


sentences = CSVSentencesToIterator("../scored_cleaned_data.csv")

model = gensim.models.Word2Vec(sentences, iter=30, size=60, window=5, min_count=10 )
print(model)

model.save("models/1000w_WordsVectorModel_size50_window5_mincount_10.model")

for e in model.most_similar(positive=['垃圾'], topn=10):
   print(e[0], e[1])







