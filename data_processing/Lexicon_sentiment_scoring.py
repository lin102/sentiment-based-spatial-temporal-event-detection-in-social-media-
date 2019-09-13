import jieba
import pandas as pd
import time
import jieba.posseg as pseg
import os

"""
    Load All the dictionaries
"""

sentiment_words_path = "../dictionaries/sentiment_dictionaries/DalianTU.csv"
stop_words_path = "../dictionaries/stopword_dictionaries/Harbin.txt"
degree_words_path = "../dictionaries/degree_list.csv"
negative_prefix_path = "../dictionaries/negative_prefix_list.txt"

# Load stop word list
stopWord_list = []
with open(stop_words_path, "r") as stopWord:
    for line in stopWord:
        line = line.strip()
        stopWord_list.append(line)
    # the way below is faster
    # stopWord_list = stopWord.read().splitlines()
# print(stopWord_list)

# Load sentiment word list
sentiment_words = pd.read_csv(sentiment_words_path, encoding='utf-8')

# Load Degree words list
degree_words = pd.read_csv(degree_words_path, encoding='utf-8', header=None, names=["word","score"])
# print(degree_words)

# Load negative prefix list
negative_prefix_list = []
with open(negative_prefix_path, "r") as negativeWords:
    for line in negativeWords:
        line = line.strip()
        negative_prefix_list.append(line)
# print(negative_prefix_list)


"""
   The Core Scoring Function
"""


def scoring(text):

    """
    Give each microblog a sentiment score
    >0 means it is a positive sentiment 
    <0 means it is a negative sentiment 
    0 means it does not show obvious sentiment
    """

    seg_list = jieba.cut(text)
    #print(type(seg_list))

    score = 0
    sentimentUnit = []

    for seg in seg_list:
        #print(sentimentUnit)
        # Filter out all the stop words
        if seg in stopWord_list:
            #print(seg)
            continue
        else:
            sentimentUnit.append(seg)

        # check if this word in the sentiment list
        if not sentiment_words[sentiment_words["word"] == seg].empty:
            # If this is a sentiment word, get the polarity of this word
            polarity = sentiment_words[sentiment_words["word"] == seg]["polarity"].iloc[0]
            #print("the word is ", seg)
            #print("The polarity of this word is ", polarity)

            # check are there degree words or negative prefix before this sentiment words
            degree = 1
            for word in sentimentUnit:

                # Check if it is a degree word
                if not degree_words[degree_words["word"] == word].empty:
                    degree = degree * degree_words[degree_words["word"] == word]["score"].iloc[0]
                    #print("The degree is ", degree)

                # check if it is a negative prefix
                if word in negative_prefix_list:
                    degree = degree * -1

            # Score for this sentiment Unit and put this score in the sum score
            score = score + degree * polarity
            #print("the score is ", score)
            # clear the sentiment unit for starting saving next sentiment unit
            sentimentUnit.clear()

    #print("This is the final score for the sentence: ", score)
    return score


path = "../weibo_senti_100k.csv"
savePath = "../scored_weibo_senti_100k.csv"
# set chunkSize and iterate loading the data preventing low memory problem
chunkSize = 10000
counter = 0
row_counter = 0

records = pd.read_csv(path, encoding='utf-8', chunksize=chunkSize)
print("Loading")


for chunk in records:
    # Iterate records

    counter = counter + 1
    print("This is the " + str(counter) + " chunk")

    # store all the empty msgText row index for deleting later
    # using iterator to Iterate rows
    for index, row in chunk.iterrows():
        msgText = chunk.loc[index, 'review']
        # in some case the msgText is NULL then the type is float
        score = scoring(msgText)
        chunk.loc[index, 'dic_score'] = score
        row_counter = row_counter + 1
        if row_counter % 2000 == 0:
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

