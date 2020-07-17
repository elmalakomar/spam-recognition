from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
import os


if __name__ == '__main__':
    # load data as raw matrix
    data = pd.read_csv("spambase.data").to_numpy()
    #shuffle train and test
    np.random.shuffle(data)
    print(data.shape)
    X=data[:,:48]  #train set
    Y=data[:,-1] # test set

    Xtrain = X[:-100,]
    Ytrain = Y[:-100,]

    Xtest = X[-100:,]
    Ytest = Y[-100:,]

    model = MultinomialNB()
    model.fit(Xtrain,Ytrain)
    print("Classification rate for NB:", model.score(Xtest,Ytest))

    model = AdaBoostClassifier()
    model.fit(Xtrain, Ytrain)
    print("Classification rate for Adaboost:", model.score(Xtest, Ytest))
    pred = model.predict(Xtest)
    score = metrics.accuracy_score(Ytest, pred)
    print("accuracy:   %0.3f" % score)

    print("classification report:")
    print(metrics.classification_report(Ytest, pred,target_names=["not_spam","spam"]))

    print("confusion matrix:")
    print(metrics.confusion_matrix(Ytest, pred))
