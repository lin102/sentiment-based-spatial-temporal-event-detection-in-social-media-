import os
import jieba
import numpy as np
import pandas as pd
import gensim
import joblib


# return the average vector for representation of this sentence
def sentence_scoring(text, vector_model, stop_word_list):

    score = None
    # text segmentation
    seg_list = jieba.cut(text)
    # list for storing all the word vectors in the sentences
    vector_list = []

    for seg in seg_list:
        # print(sentimentUnit)
        # Filter out all the stop words
        if seg in stop_word_list:
            # print(seg)
            continue
        else:
            try:
                # using trained word2vec model convert the word into vector, return value is a numpy vector
                vector = vector_model.wv[seg]
                vector_list.append(vector)
            except KeyError:
                continue

    # if the sentence can not be vectorized
    if len(vector_list) == 0:
        # -99 is the value for sentences which can not be vectorized or has no sentiment
        score = -99

    else:
        # get the average vector for presenting the whole centence
        average_vector = sum(vector_list) / len(vector_list)
        score = clf.predict([average_vector])[0]

    return score


# -----------------------Load stop words list--------------------------------------

stop_words_path = "../dictionaries/stopword_dictionaries/Harbin.txt"
stopWord_list = []
with open(stop_words_path, "r") as stopWord:
    for line in stopWord:
        line = line.strip()
        stopWord_list.append(line)
    # the way below is faster
    # stopWord_list = stopWord.read().splitlines()

# ----------------------- Load trained Word2Vec model-----------------------------

vector_model = gensim.models.Word2Vec.load('models/zh.bin')
vector_model.score()

# ------------------------ Load trained-yet classifor--------------------------------

clf = joblib.load("trained_classification_models/logistics_regression_model.joblib")


# ----------------load data--------------------------------------------------------

path = "../cleaned_data.csv"
savePath = "../scored_cleaned_data_M_log.csv"
# set chunkSize and iterate loading the data preventing low memory problem
chunkSize = 10000
chunk_counter = 0
records = pd.read_csv(path, encoding='utf-8', chunksize=chunkSize)
print("Loading")

for chunk in records:
    # Iterate records

    chunk_counter = chunk_counter + 1
    print("This is the " + str(chunk_counter) + " chunk")

    chunk['M_log_score'] = chunk['msgtext'].apply(sentence_scoring, vector_model=vector_model, stop_word_list=stopWord_list)

    # check if the file is already excite otherwise create one
    if not os.path.isfile(savePath):
        chunk.to_csv(savePath, encoding='utf-8',index=False)
        print("Create a new File")
    else:
        # if the file excites append the rest chunks on the file with no header
        chunk.to_csv(savePath, encoding='utf-8', mode='a', header=False, index=False)
        print("Append on the File")

print("Scoring finished")
print("CSV File exported!")

