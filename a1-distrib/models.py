# models.py

from sentiment_data import *
from utils import *

from collections import Counter
import re
import nltk
from nltk.corpus import stopwords 
import numpy as np
from tqdm import tqdm
import random
import string
random.seed(0)


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """

    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> List[int]:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")
       
class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    #you could add new properties
    def __init__(self, indexer: Indexer):
        #FeatureExtractor.__init__(self)
        self.indexer = indexer

    def get_indexer(self): 
        return self.indexer
    
    def add_features(self, sentence: List[str]):
        words = []
        for word in sentence: 
            word = word.lower()
            if word not in words: 
                self.indexer.add_and_get_index(word)

    def extract_features(self, sentence: List[str], add_to_indexer: bool= False) -> List[int]: 
        if add_to_indexer: 
            self.add_features(sentence)
        count = Counter()
        for word in sentence: 
            word = word.lower()
            if self.indexer.contains(word): 
                index = self.indexer.index_of(word)
                count.update([index]) 
        
        return list(count.items())
    
    def vocab_size(self): 
        return len(self.indexer)


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
    
    def get_indexer(self): 
        return self.indexer

    def add_features(self, sentence: List[str]): 
        for i in range(len(sentence)-1): 
            word_pair = sentence[i].lower() + sentence[i+1].lower()
            self.indexer.add_and_get_index(word_pair)

    def extract_features(self, sentence:List[str], add_to_indexer: bool=False)-> List[int]: 
        if add_to_indexer: 
            self.add_features(sentence)
        count = Counter()
        words = []
        # for word in sentence: 
        #     if word not in string.punctuation: 
        #          words.append(word) 

        for i in range(len(sentence)-1): 
            word_pair = sentence[i].lower() + sentence[i+1].lower()
             
            if self.indexer.contains(word_pair): 
                index = self.indexer.index_of(word_pair)
                count.update([index])

        return list(count.items())
    
    def vocab_size(self): 
        return len(self.indexer)

class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        #raise Exception("Must be implemented")
        #FeatureExtractor.__init__(self)
        self.indexer = indexer
    
    def get_indexer(self): 
        return self.indexer
    
    def add_features(self, sentence: List[str]): 
        #remove punctation and stop words
        words = []
        stop_words = set(stopwords.words('english')) 
        for word in sentence: 
            if word not in string.punctuation: 
                word = word.lower()
                words.append(word)
        
        non_stop_words = [w for w in words if not w in stop_words]
        for feature in non_stop_words: 
            self.indexer.add_and_get_index(feature)
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool= False)-> List[int]: 
        count = Counter()
        if add_to_indexer: 
            self.add_features(sentence)
        for word in sentence: 
            word = word.lower()
            if self.indexer.contains(word): 
                index = self.indexer.index_of(word)
                count.update([index]) 
        
        return list(count.items())

    def vocab_size(self): 
        return len(self.indexer)

class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, feat_extractor):
        #raise Exception("Must be implemented")
        SentimentClassifier.__init__(self)
        self.feat_extractor = feat_extractor
        self.indexer = self.feat_extractor.get_indexer()
        self.words_size = self.feat_extractor.vocab_size()
        self.weights = np.zeros((self.words_size,))
        self.feature_dic = {}
    
    def get_features(self, sentence: List[str]) -> List[int]: 
        ex_sent = ''.join(sentence)
        if ex_sent not in self.feature_dic: 
            feature = self.feat_extractor.extract_features(sentence)
            self.feature_dic[ex_sent] = feature
        else: 
            feature = self.feature_dic[ex_sent]

        return feature

    def predict(self, sentence: List[str]) -> int: 
        feature = self.get_features(sentence)
        weight_multi_feature = 0
        for key,val in feature:
            weight_multi_feature += self.weights[key] * val
        y_pre = 1 if weight_multi_feature >=0.5 else 0
        
        return y_pre

    def update_weight(self, sentence, y, y_pre, alpha): 
        features = self.get_features(sentence)
        for k, val in features: 
            self.weights[k] = self.weights[k] - (y_pre - y) * alpha * val

#Calculate sigmoid
def sigmoid(x):
    result = 1./(1. + np.exp(-x))
    return result

class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, feat_extractor):
        #raise Exception("Must be implemented")
        SentimentClassifier.__init__(self)
        self.feat_extractor = feat_extractor
        self.indexer = self.feat_extractor.get_indexer()
        self.words_size = self.feat_extractor.vocab_size()
        self.weights = np.zeros((self.words_size,))
        self.feature_dic = {}

    def get_features(self, sentence: List[str]) -> List[int]:
        ex_sent = ''.join(sentence)
        if ex_sent not in self.feature_dic: 
            feature = self.feat_extractor.extract_features(sentence)
            self.feature_dic[ex_sent] = feature
        else: 
            feature = self.feature_dic[ex_sent]

        return feature

    def predict(self, sentence: List[str]) -> int: 
        features = self.get_features(sentence)
        weight_multi_feature = 0 
        for k , val in features: 
            weight_multi_feature += self.weights[k] * val

        result = sigmoid(weight_multi_feature)
        y_pre = 1 if result>0.5 else 0

        return y_pre

    def update_weight(self, sentence, y, y_pre, alpha): 
        features = self.get_features(sentence)

        weight_multi_feature = 0
        for k ,val in features:
            weight_multi_feature += self.weights[k] * val
        
        result = sigmoid(weight_multi_feature)
        for k , val in features:
            self.weights[k] = self.weights[k] - alpha * ((result - y)* val)
            #self.weights[k] = self.weights[k] - alpha * (-y * val * (1-result) + (1-result) * val * result)

    def loss(self, train_exs): 
        sum_loss = 0
        for ex in train_exs: 
            x = ex.words
            y = ex.label
            feature = self.get_features(ex.words)  
        
        weight_multi_feature = 0
        for k , val in feature: 
            weight_multi_feature += self.weights[k] * val
        
        result = sigmoid(weight_multi_feature)
        loss = -y * np.log(result) - (1-y)* np.log(1-result)
        sum_loss += loss
        sum_loss = sum_loss / float(len(train_exs))

        return sum_loss


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    #raise Exception("Must be implemented")
    #extract all the features first: 
    for ex in train_exs: 
        feat_extractor.add_features(ex.words)

    model = PerceptronClassifier(feat_extractor)
    epochs = 20
    alpha = 1
    for i in tqdm(range(epochs)): 
        random.shuffle(train_exs)
        data_size = int(len(train_exs))
        data_ex = train_exs[:data_size]

        for ex in data_ex: 
            y = ex.label
            y_pre = model.predict(ex.words)
            model.update_weight(ex.words, y, y_pre, alpha)
        
        alpha = alpha * 0.9
    
    return model

def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    #raise Exception("Must be implemented")
    for ex in train_exs: 
        feat_extractor.add_features(ex.words)

    model = LogisticRegressionClassifier(feat_extractor)
    epochs = 30
    alpha = 0.5
   
    for i in tqdm(range(epochs)): 
        """ if(isinstance(feat_extractor, BetterFeatureExtractor)): 
            alpha = alpha / (i+1) """
        random.shuffle(train_exs)
        data_size = int(len(train_exs))
        data_exs = train_exs[:data_size]

        for ex in data_exs: 
            y = ex.label
            y_pre = model.predict(ex.words)
            model.update_weight(ex.words, y, y_pre, alpha)
        
    return model

#do not modify
def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model