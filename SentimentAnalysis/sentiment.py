import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import random
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
import sys
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
import statistics
import pandas as pd
import gc

class PreProcessing():

    def check_balance(self, pos, neg):
        return len(pos) == len(neg)

    def delete_missing(self, data):
        return data.dropna()

    def get_columns(self, columns, data):
        return data.loc[:, columns]


class classifier(PreProcessing):

    def __init__(self, trained=False, stopwf=False):
        self.all_words = []
        self.tweetTokenized = []
        self.word_features = []
        self.featuresets = []
        self.stopWF = stopwf
        self.training_set = []
        self.testing_set = []
        if not trained:
            self.set_training()
            tamano = round(len(self.featuresets) * 0.8)
            self.training_set = self.featuresets[:tamano]
            self.testing_set = self.featuresets[tamano:]
            print("set de entrenamiento tiene " + str(len(self.training_set)), " features")
            print("set de evaluacion tiene " + str(len(self.testing_set)), " features")
            self.train()
        else:
            pass



    def set_training(self):
        print("leyendo el set de datos para entrenar ...")
        self.data = pd.read_csv('tweets_dataset.csv',
                                names=['Target', 'id', 'Date', 'QUERY', 'User', 'Tweet'],
                                encoding='latin-1', engine='python')
        print("Set de datos cargado. Haciendo preprocesamiento ...")




        self.data = PreProcessing.get_columns(self, ['Target', 'Tweet'], self.data)
        self.data = PreProcessing.delete_missing(self, self.data)
        self.data = self.data.sample(frac=0.10)

        self.data = self.data.values.tolist()
        print(len(self.data))
        '''for i in range(300000):
            sys.stdout.write("\rtweets eliminados = %i" % i)
            sys.stdout.flush()
            del self.data[i]
            del self.data[-i-1]
        print("\n"+str(len(self.data)))'''
        print("Se procede a tokenizar todos los tweets\n")
        i = 0
        for phrase in self.data:
            i += 1
            sys.stdout.write("\rtweets tokenizados = %i" % i)
            sys.stdout.flush()
            tokenized = word_tokenize(phrase[1])
            for w in tokenized:
                self.all_words.append(w.lower())
        '''i = 0
        print("\n")
        for phrase in self.data[self.data['Target'] == 4]['Tweet']:
            i += 1
            sys.stdout.write("\rFrases positivas analizadas = %i" % i)
            sys.stdout.flush()
            tokenized = word_tokenize(phrase)
            for w in tokenized:
                self.all_words.append(w.lower())
'''

        self.all_words = nltk.FreqDist(self.all_words)
        self.word_features = list(self.all_words.keys())[:1000]
        gc.collect()
        i = 0
        '''self.featuresets = ((self.find_features(tweet), label) for label, tweet in self.data)
        self.featuresets = list(self.featuresets)'''
        print("\nFeature set empezando\n")
        for (label, tweet) in self.data:
            i += 1
            sys.stdout.write("\rFeature Set size = %i" % i)
            sys.stdout.flush()
            if label == 4:
                self.featuresets.append((self.find_features(tweet), 1))
            else:
                self.featuresets.append((self.find_features(tweet), label))
        random.shuffle(self.featuresets)
        print("\n")
        print(len(self.featuresets))
        print("Modelo listo para entrenar")

        print('Guardando data en pickle')
        save_documents = open("pickled_algos/documents.pickle", "wb")
        pickle.dump(self.data, save_documents)
        save_documents.close()
        print('Guardando word_features en picke')
        save_word_features = open("pickled_algos/word_features1k.pickle", "wb")
        pickle.dump(self.word_features, save_word_features)
        save_word_features.close()



    def find_features(self, document):
        words = word_tokenize(document)
        features = {}
        for w in self.word_features:
            features[w] = (w in words)

        return features

    def train(self):
        gc.collect()

        print("Empezando el entrenamiento\n")
        classifier = nltk.NaiveBayesClassifier.train(self.training_set)
        print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, self.testing_set)) * 100)
        classifier.show_most_informative_features(15)
        gc.collect()
        save_classifier = open("pickled_algos/originalnaivebayes1k.pickle", "wb")
        pickle.dump(classifier, save_classifier)
        save_classifier.close()

        MNB_classifier = SklearnClassifier(MultinomialNB())
        MNB_classifier.train(self.training_set)
        print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, self.testing_set)) * 100)
        gc.collect()
        save_classifier = open("pickled_algos/MNB_classifier1k.pickle", "wb")
        pickle.dump(MNB_classifier, save_classifier)
        save_classifier.close()

        BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
        BernoulliNB_classifier.train(self.training_set)
        print("BernoulliNB_classifier accuracy percent:",
              (nltk.classify.accuracy(BernoulliNB_classifier, self.testing_set)) * 100)
        gc.collect()
        save_classifier = open("pickled_algos/BernoulliNB_classifier1k.pickle", "wb")
        pickle.dump(BernoulliNB_classifier, save_classifier)
        save_classifier.close()

        SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
        SGDClassifier_classifier.train(self.training_set)
        print("SGDClassifier_classifier accuracy percent:",
              (nltk.classify.accuracy(SGDClassifier_classifier, self.testing_set)) * 100)
        gc.collect()
        save_classifier = open("pickled_algos/SGDClassifier_classifier1k.pickle", "wb")
        pickle.dump(SGDClassifier_classifier, save_classifier)
        save_classifier.close()

        LinearSVC_classifier = SklearnClassifier(LinearSVC())
        LinearSVC_classifier.train(self.training_set)
        print("LinearSVC_classifier accuracy percent:",
              (nltk.classify.accuracy(LinearSVC_classifier, self.testing_set)) * 100)
        gc.collect()
        save_classifier = open("pickled_algos/LinearSVC_classifier1k.pickle", "wb")
        pickle.dump(LinearSVC_classifier, save_classifier)
        save_classifier.close()

classifier()
