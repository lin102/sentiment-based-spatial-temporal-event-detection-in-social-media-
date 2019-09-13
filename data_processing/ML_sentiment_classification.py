import jieba
import numpy as np
import pandas as pd
import gensim
import sklearn.naive_bayes
import sklearn
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree, svm, naive_bayes,neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegressionCV
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score
from sklearn import metrics



# return the average vector for representation of this sentence
def sentence_to_vector(text, vector_model, stop_word_list):

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
        # only set this to "" can be recognized later
        average_vector = np.asarray([])

    else:
        # get the average vector for presenting the whole centence
        average_vector = sum(vector_list) / len(vector_list)
        # numpy.ndarray
    return average_vector


def dataset_to_vectors(dataframe, column_name):

    # Load stop words list
    stop_words_path = "../dictionaries/stopword_dictionaries/Harbin.txt"

    # Load stop word list
    stopWord_list = []
    with open(stop_words_path, "r") as stopWord:
        for line in stopWord:
            line = line.strip()
            stopWord_list.append(line)
        # the way below is faster
        # stopWord_list = stopWord.read().splitlines()

    # load trained Word2Vec model
    vector_model = gensim.models.Word2Vec.load('models/zh.bin')
    # for a vector or serious which is 1d use python built in map is better than pandas dataframe.apply
    dataframe[column_name] = dataframe[column_name].apply(sentence_to_vector, vector_model = vector_model, stop_word_list = stopWord_list )

    # filter all the rows that can not be vectorized
    # In this case it is hard to using [] filter array so write a condition function return a True and False array on my own
    empty_item = np.asarray([])
    dataframe = dataframe[dataframe[column_name].apply(lambda x: True if(x != empty_item) else False)]
    # dataframe.drop(dataframe[(dataframe['review'] == empty_item )].index, inplace=True)

    return dataframe

# --------------------Plot Learning Curve------------------------------------------------

def plot_learning_curve(pipeline_estimator):

    train_size, train_score, test_score = learning_curve(pipeline_estimator,train_x,train_y,
                                                         train_sizes= np.linspace(0.1, 1.0, 10), cv=5, n_jobs=1)
    train_mean = np.mean(train_score, axis=1)
    train_std = np.std(train_score, axis=1)
    test_mean = np.mean(test_score, axis=1)
    test_std = np.std(test_score, axis=1)

    # plot learning curve
    plt.plot(train_size, train_mean, color='blue', marker='o',
             markersize=5, label='training accuracy')
    plt.fill_between(train_size,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='blue')
    plt.plot(train_size, test_mean, color='green',
             linestyle ='--', marker='s', markersize=5,
             label='validation accuracy')
    plt.fill_between(train_size,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')
    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.5, 1.0])
    plt.show()


#Grid search CV
def grid_search_CV(pipeline_estimator):

    parameters = {'logistic_regression__solver': ['lbfgs', 'liblinear'],
                  'logistic_regression__C':[0.01, 0.1, 1,]}
    grid_search = GridSearchCV(pipeline_estimator, parameters, cv=5, refit=True)
    grid_search.fit(train_x, train_y)
    print('Best params: ', grid_search.best_params_)
    print('Best score: ', grid_search.best_score_)

    return grid_search


# ---------------------- Plot Validation Curve --------------------------------

def plot_validation_curve(pipeline_estimator, param_name):

    #parametersRange = [1/0.02, 1//0.04, 1/0.08, 1/0.16,1/0.32,1/0.64,1/1.28,1/2.56,1/5.12,1/10.24, 1/20.48,1/40.96, 1/81.92]
    #parametersRange = [0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92]
    #parametersRange = [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4]
    parametersRange = [10,20,50,70,100,150,200,300,400,500,600]
    train_scores, test_scores = validation_curve(pipeline_estimator, train_x, train_y,
                                                 param_name=param_name,
                                                 param_range=parametersRange,
                                                 cv=5)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # plot validation curve
    plt.plot(parametersRange, train_mean,
             color='blue', marker='o', markersize=5,
             label='Training Accuracy')
    plt.fill_between(parametersRange,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='blue')
    plt.plot(parametersRange, test_mean,
             color='green', marker='s', markersize=5,
             linestyle='--', label='Validation Accuracy')
    plt.fill_between(parametersRange,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')
    plt.grid()
    #plt.xscale('log')
    plt.legend(loc='lower right')
    plt.xlabel('Parameter C')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1.0])
    plt.show()

def build_and_save_best_model(pipeline_estimator):

    # get the best hyper parameters by GridSearch CV
    estimator = grid_search_CV(pipeline_estimator)
    print('generalization score: ',np.mean(estimator.predict(test_x) == test_y))
    best_params = estimator.best_params_
    pipeline_estimator.set_params(**best_params)
    pipeline_estimator.fit(vectorized_x,vectorized_y)

    # ---------save the model
    from joblib import dump
    # directory
    trained_classification_model_path = "trained_classification_models/logistics_regression_model.joblib"
    try:
        dump(pipeline_estimator, trained_classification_model_path)
    except Exception:
        print("Model construction failed")
    else:
        print("Model has been successfully built and saved")

def load_trained_model():

    from joblib import load
    clf = load("trained_classification_models/logistics_regression_model.joblib")
    print(np.mean(clf.predict(test_x) == test_y))


if __name__ == '__main__':

    # ----------Data loading---------------------------------------------------
    print("Loading")
    #path = "../shuffled_1000_weibo.csv"
    path = "../weibo_senti_100k.csv"
    records = pd.read_csv(path, encoding='utf-8')

    # ----------Data Vectorization and word embedding-------------------------

    vectorized_df = dataset_to_vectors(records, "review")
    vectorized_x = vectorized_df['review']
    vectorized_y = vectorized_df['label']
    # the stack make the 1d array to 2d numpy array
    vectorized_x = np.stack(np.asarray(vectorized_x))
    vectorized_y = np.asarray(vectorized_y)

    # ----------Data set division-----------------------------------------------
    # ----------Alter to Cross validation later, ------------------------------

    # divide the data set into training data set and valid data set, the proportion of test size is 20%
    train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(vectorized_x, vectorized_y, test_size=0.2)
    print("Data set Division Complete", len(train_x), len(test_x))

    # ----------Scaling and standardization---------------------------------

    scaler = preprocessing.StandardScaler()
    # scaler = preprocessing.StandardScaler().fit(train_x)
    # train_x = scaler.transform(train_x)
    # test_x = scaler.transform(test_x)
    print("Scaling Complete")

    # ----------Classifier and training--------------------------------------

    # 1. Kneighbors classifier
    knn = KNeighborsClassifier()
    # knn.fit(train_x, train_y)
    # pred = knn.predict(test_x)
    # clf = sklearn.naive_bayes.MultinomialNB().fit(train_x, train_y)

    #print(np.mean(pred == test_y))
    #print(knn.score(test_x,test_y))
    #print(pred)

    # 2. SVC
    svc_linear = svm.SVC(kernel='linear')
    svc = svm.SVC()

    # 3. Naive Bayes

    # GaussianNB
    bayes_gaussian = naive_bayes.GaussianNB()
    # Multinomial Naive Bayes
    bayes_multinomial = naive_bayes.MultinomialNB()

    # 4.Logistic Regression
    logistic_regression = LogisticRegression(C=0.01, solver='lbfgs')

    # 5. random forest classifier
    randomForestClf = RandomForestClassifier()

    # ----------Pipeline---------------------------------------------------------

    print("--------------------------------- KNN ------------------------------")
    # pipeline_knn = Pipeline([('scaler', scaler),
    #                       ('knn', knn),
    #                       ])
    #
    # print(pipeline_knn)
    # # pipeline_knn.fit(train_x, train_y)
    # #
    # # plot_learning_curve(pipeline_knn)
    # #
    # # scores = cross_val_score(pipeline_knn, vectorized_x, vectorized_y, cv=5, n_jobs=-1)
    #
    # accuray_scores = cross_val_score(pipeline_knn, vectorized_x, vectorized_y, cv=5)
    # print('accuracy: ', accuray_scores.mean(), accuray_scores.std() * 2)
    # f1_score = cross_val_score(pipeline_knn, vectorized_x, vectorized_y, cv=5, scoring='f1')
    # print('f1: ', f1_score.mean())



    print("------------------------------- SVM with linear kernel--------------------------------")
    # pipeline_svc_linear = Pipeline([('scaler', scaler),
    #                          ('svc_linear', svc_linear),
    #                          ])
    #
    # print(pipeline_svc_linear)
    # # pipeline_svc_linear.fit(train_x, train_y)
    # # print(np.mean(pipeline_svc_linear.predict(test_x) == test_y))
    #
    # accuray_scores = cross_val_score(pipeline_svc_linear, vectorized_x, vectorized_y, cv=5)
    # print('accuracy: ', accuray_scores.mean(), accuray_scores.std() * 2)
    # f1_score = cross_val_score(pipeline_svc_linear, vectorized_x, vectorized_y, cv=5, scoring='f1')
    # print('f1: ', f1_score.mean())
    #
    print("------------------------------- pipeline_bayes_gaussian --------------------------------")
    # pipeline_bayes_gaussian = Pipeline([('scaler', scaler),
    #                                 ('bayes_gaussian', bayes_gaussian),
    #                                 ])
    #
    # print(pipeline_bayes_gaussian)
    # # pipeline_bayes_gaussian.fit(train_x, train_y)
    # # print(np.mean(pipeline_bayes_gaussian.predict(test_x) == test_y))
    #
    # accuray_scores = cross_val_score(pipeline_bayes_gaussian, vectorized_x, vectorized_y, cv=5)
    # print('accuracy: ', accuray_scores.mean(), accuray_scores.std() * 2)
    # f1_score = cross_val_score(pipeline_bayes_gaussian, vectorized_x, vectorized_y, cv=5, scoring='f1')
    # print('f1: ', f1_score.mean())

    print("--------------------------------- bayes_multinomial ------------------------------")

    # pipeline_bayes_multinomial = Pipeline([('scaler', scaler),
    #                                 ('bayes_multinomial', bayes_multinomial),
    #                                 ])
    #
    # print(pipeline_bayes_multinomial)
    # # pipeline_bayes_multinomial.fit(train_x, train_y)
    # # print(np.mean(pipeline_bayes_multinomial.predict(test_x) == test_y))
    # pipeline_knn.fit(train_x, train_y)
    #
    # plot_learning_curve(pipeline_knn)
    #
    # scores = cross_val_score(pipeline_knn, vectorized_x, vectorized_y, cv=5, n_jobs=-1)
    #
    # accuray_scores = cross_val_score(pipeline_bayes_multinomial, vectorized_x, vectorized_y, cv=5)
    # print('accuracy: ', accuray_scores.mean(), accuray_scores.std() * 2)
    # f1_score = cross_val_score(pipeline_bayes_multinomial, vectorized_x, vectorized_y, cv=5, scoring='f1')
    # print('f1: ', f1_score.mean())
    #
    # precison_score = cross_val_score(pipeline_knn, vectorized_x, vectorized_y, cv=5,
    #                                  scoring='precision')
    # print('Precision: ', precison_score.mean())
    # recall_score = cross_val_score(pipeline_knn, vectorized_x, vectorized_y, cv=5, scoring='recall')
    # print('Recall: ', recall_score.mean())


    print("--------------------------------- logistic_regression ------------------------------")

    # pipeline_logistic_regression = Pipeline([('scaler', scaler),
    #                                 ('logistic_regression', logistic_regression),
    #                                 ])
    #
    # print(pipeline_logistic_regression)
    # pipeline_logistic_regression.fit(train_x, train_y)
    # #print(np.mean(pipeline_logistic_regression.predict(test_x) == test_y))
    #
    # accuray_scores = cross_val_score(pipeline_logistic_regression, vectorized_x, vectorized_y, cv=5)
    # print('accuracy: ',accuray_scores.mean(), accuray_scores.std() * 2)
    # f1_score = cross_val_score(pipeline_logistic_regression, vectorized_x, vectorized_y, cv=5, scoring='f1')
    # print('f1: ',f1_score.mean())
    # precison_score = cross_val_score(pipeline_logistic_regression, vectorized_x, vectorized_y, cv=5, scoring='precision')
    # print('Precision: ', precison_score.mean())
    # recall_score = cross_val_score(pipeline_logistic_regression, vectorized_x, vectorized_y, cv=5, scoring='recall')
    # print('Recall: ', recall_score.mean())

    print("--------------------------------- Random Forest ------------------------------")

    # pipeline_random_forest = Pipeline([('scaler', scaler),
    #                                 ('random_forest', randomForestClf),
    #                                 ])
    #
    # print(pipeline_random_forest)
    # # pipeline_random_forest.fit(train_x, train_y)
    # # print(np.mean(pipeline_random_forest.predict(test_x) == test_y))
    # # print(pipeline_random_forest.score(test_x, test_y))
    # # plot_validation_curve(pipeline_random_forest,'random_forest__n_estimators')
    #
    # accuray_scores = cross_val_score(pipeline_random_forest, vectorized_x, vectorized_y, cv=5)
    # print('accuracy: ', accuray_scores.mean(), accuray_scores.std() * 2)
    # f1_score = cross_val_score(pipeline_random_forest, vectorized_x, vectorized_y, cv=5, scoring='f1')
    # print('f1: ', f1_score.mean())








