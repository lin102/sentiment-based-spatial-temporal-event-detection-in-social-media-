from wordcloud import WordCloud, ImageColorGenerator
import pandas as pd
from collections import defaultdict
import jieba
import jieba.analyse
import matplotlib.pyplot as plt


# this function is for pandas apply() iterate every row
def term_frequency_calculation(text, tf_dict: defaultdict, stopword_list):
    for x in jieba.analyse.extract_tags(text, topK=7, allowPOS=part_of_speech):
        if x in stopword_list:
            continue
        else:
            tf_dict[x] = tf_dict[x] + 1


def words_filter_based_on_frequency(dictionary, low_frequency, high_frequency):

    # filter the words based on the frequency
    # different usages for positive and negative event detection,
    # positive event detection does not need to filter, negative event need to filter high frequency words
    # because the most high frequency words are still related to the positive event

    for key in list(dictionary):
        if dictionary[key] < low_frequency or dictionary[key] > high_frequency:
            del dictionary[key]
    return dictionary


def draw_words_cloud(dic, imag_save_path):
    word_cloud = WordCloud(font_path='../chinese_font/song.ttf',background_color='white',max_words=50,max_font_size=60).generate_from_frequencies(dic)
    plt.imshow(word_cloud)
    plt.axis("off")
    plt.show()
    word_cloud.to_file(imag_save_path)


if __name__ == '__main__':

    # all the dictionaries path
    stop_words_path = "../dictionaries/stopword_dictionaries/Harbin.txt"

    # the part of speech list tuple for the jieba to filter adj words
    part_of_speech = (
    'b', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'an', 'i', 'j', 'k', 'l', 'Ng', 'n', 'nr', 'ns', 'nt', 'nz', 'r', 's', 'tg',
    't', 'u', 'v', 'vg', 'vd', 'vn', 'un', 'z')

    # Load stop word list
    stop_word_list = []
    with open(stop_words_path, "r") as stopWord:
        for line in stopWord:
            line = line.strip()
            stop_word_list.append(line)
    # this dict is for storing all the words and their frequency，default value is 0

    # the event path
    read_path = '../Mercedes_Benz_Arena_20140511_negative.csv'
    words_cloud_img_save_path = "../word_cloud_img/20140511_benz.jpg"
    # store all the high frequency words
    term_frequency_dict = defaultdict(lambda: 0)
    # read event file
    records = pd.read_csv(read_path,encoding='utf-8')
    # Using TF-IDF to calculate all the key words and put them into the dict
    records['msgtext'].apply(term_frequency_calculation, tf_dict=term_frequency_dict, stopword_list=stop_word_list)

    # filter out all the low frequency words
    # this is used for negative event detection
    #term_frequency_dict = words_filter_based_on_frequency(term_frequency_dict, 10, 500)

    print(term_frequency_dict)
    # draw the words cloud
    draw_words_cloud(term_frequency_dict,words_cloud_img_save_path)
