# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *
from tqdm import tqdm
import torch.nn.functional as F

class FFNN(nn.Module):
    def __init__(self, word_embeddings=None, inp=50, hid=32, out=2):
        super(FFNN, self).__init__()
        if word_embeddings is not None:
            vocab = len(word_embeddings.vectors)
            self.embeddings =nn.Embedding.from_pretrained(torch.from_numpy(word_embeddings.vectors), freeze=False)
        self.V = nn.Linear(inp, hid)
        self.g = nn.Tanh()
        #self.g = nn.ReLU()
        self.W = nn.Linear(hid, out)
        #self.log_softmax = nn.LogSoftmax(dim=0)
        # Initialize weights according to a formula due to Xavier Glorot.
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, x):
        if self.embeddings is not None :
            word_embedding = self.embeddings(x) 
            mean = torch.mean(word_embedding, dim=1, keepdim=False).float()
            return self.W(self.g(self.V(mean)))
        else:
            return self.W(self.g(self.V(x)))


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, word_embeddings: WordEmbeddings):
        SentimentClassifier.__init__(self)
        self.word_indexer = word_embeddings.word_indexer
        self.input = word_embeddings.get_embedding_length()
        self.hidden= 32
        self.output= 2
        self.loss = nn.CrossEntropyLoss()
        self.model = FFNN(word_embeddings, self.input, self.hidden, self.output)

    def predict(self, ex_words: List[str]):
        words_idx = [max(1, self.word_indexer.index_of(word)) for word in ex_words]
        words_tensor=torch.tensor([words_idx])
        y_probs = self.model.forward(words_tensor)
        return torch.argmax(y_probs)

    def loss(self, probs, target):
        return self.loss(probs, target)


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    ns_classifier = NeuralSentimentClassifier(word_embeddings)
    word_indices = {}

    for i in range(len(train_exs)):
        words = train_exs[i].words
        index_list = []
        for word in words:
            idx = ns_classifier.word_indexer.index_of(word)
            index_list.append(max(idx, 1))
        word_indices[i] = index_list

    epochs = 15
    learning_rate = 0.001
    batch_size= 128
    optimizer = optim.Adam(ns_classifier.model.parameters(), lr=learning_rate)

    ex_indices = [idx for idx in range(0,len(train_exs))]
    
    for epoch in tqdm(range(epochs)):
        random.shuffle(ex_indices)
        total_loss = 0.0
        batch_x = []
        batch_y = []
        pad_length=50
        for idx in ex_indices:
            if len(batch_x)<batch_size:
                sent_pad = [0]*pad_length
                sent = word_indices[idx]
                # padding
                sent_pad[:min(pad_length,len(sent))]=sent[:min(pad_length,len(sent))]
                batch_x.append(sent_pad)
                y = train_exs[idx].label
                batch_y.append(y)

            else:   # len(batch_x) = batch_size
                ns_classifier.model.train()
                optimizer.zero_grad()
                batch_x = torch.tensor(batch_x)
                probs =  ns_classifier.model.forward(batch_x)
                target = torch.tensor(batch_y)
                loss =  ns_classifier.loss(probs, target)
                total_loss += loss
                
                loss.backward()
                
                optimizer.step()
                batch_x = []
                batch_y = []

        total_loss /= len(train_exs)
        print("Total loss on epoch %i: %f" % (epoch, total_loss))
    
    return ns_classifier

