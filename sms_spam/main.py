import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
import scipy.sparse as sp
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def visualizeWordCloud(label):
    #get all words from spam msgs
    words = ''
    for msg in data[data['labels'] == label]['data']:
        msg = msg.lower()
        words += msg + ' '
    wordcloud = WordCloud(width=600, height=400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':

    #1. Load the data
    data = pd.read_csv("spam.csv", engine='python')
    ## or
    # data = pd.read_csv("spam.csv", encoding='ISO-8859-1')

    data = shuffle(data)
    #2. process the data

    # drop unnecessary columns
    data = data.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis = 1)
    print(data.shape)

    #rename columns because are not very descriptive
    data.columns = ['labels','data']

    # get input
    X = data["data"]
    print("Input X: {}".format(X.shape))

    # compute the feature vector of X.
    vectorizer = TfidfVectorizer()
    X1 = vectorizer.fit_transform(X)
    vectorizer = CountVectorizer()
    X2 = vectorizer.fit_transform(X)
    # I concatenated Tf-idf and raw count
    X = sp.hstack((X1, X2), format='csr')
    print("Input X after preprocessing: {}".format(X.shape))


    Y = data["labels"]
    le = LabelEncoder()
    le.fit(["ham","spam"])
    Y = le.transform(Y)
    ## could alse be done as
    # Y = Y.map({'ham':0,'spam':1})
    print(Y.shape)

    # Xtrain = X[:-500, ] #prendo tutti tranne gli ultimi 500
    # print(Xtrain.shape)
    # Ytrain = Y[:-500, ]
    # print(Ytrain.shape)
    #
    # Xtest = X[-500:, ] #prendo gli ultimi 500
    # print(Xtest.shape)
    # Ytest = Y[-500:, ]
    # print(Ytest.shape)

    #could alse be done as
    Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y, test_size=0.33)

    # apply ML on the data
    model = MultinomialNB()
    model.fit(Xtrain,Ytrain)

    # evaluate
    print("Classification rate for NB:", model.score(Xtest,Ytest))

    pred = model.predict(Xtest)
    score = metrics.accuracy_score(Ytest, pred)
    print("accuracy:   %0.3f" % score)

    print("confusion matrix:")
    print(metrics.confusion_matrix(Ytest, pred))

    print("classification report:")
    print(metrics.classification_report(Ytest, pred, target_names=["not_spam", "spam"]))

    visualizeWordCloud('spam')

