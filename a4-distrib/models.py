# models.py

import torch
import torch.nn as nn
import numpy as np
import collections 
from torch import optim
import random
import time
from torch.utils.data import DataLoader
#####################
# MODELS FOR PART 1 #
#####################

class ConsonantVowelClassifier(object):
    def predict(self, context):
        """
        :param context:
        :return: 1 if vowel, 0 if consonant
        """
        raise Exception("Only implemented in subclasses")


class FrequencyBasedClassifier(ConsonantVowelClassifier):
    """
    Classifier based on the last letter before the space. If it has occurred with more consonants than vowels,
    classify as consonant, otherwise as vowel.
    """
    def __init__(self, consonant_counts, vowel_counts):
        self.consonant_counts = consonant_counts
        self.vowel_counts = vowel_counts

    def predict(self, context):
        # Look two back to find the letter before the space
        if self.consonant_counts[context[-1]] > self.vowel_counts[context[-1]]:
            return 0
        else:
            return 1

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim = 64, hidden_dim = 16, num_classes = 2):
        super(RNN, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=1, batch_first= True)
        self.linear = nn.Linear(hidden_dim, num_classes)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        embedding = self.embeddings(x) 
        output, h_n = self.gru(embedding) 
        h_n = h_n.squeeze() 
        logits = self.linear(h_n)
        return logits

class RNNClassifier(ConsonantVowelClassifier):
    def __init__(self, vocab_indexer):
        super(RNNClassifier,self).__init__()
        self.indexer = vocab_indexer
        self.vocab_size = len(vocab_indexer)
        self.model = RNN(self.vocab_size)
        self.loss = nn.CrossEntropyLoss()
        
    def predict(self, context):
        context_list = list(context)
        context_list_idx = [self.indexer.index_of(i) for i in context_list]
        context_list_tensor = torch.tensor([context_list_idx])
        probs = self.model(context_list_tensor)
        return torch.argmax(probs)

        
def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)

#from raw string to a pytorch tensor of indices
class ProcessData(torch.utils.data.Dataset):
    def __init__(self, cons, vowels, indexer):
        super(ProcessData, self).__init__()
        self.cons_data = cons
        self.vowel_data = vowels
        self.indexer = indexer
        self.preprocess()

    def preprocess(self):
        self.characters = []
        self.labels = []
        for cons in self.cons_data:
            cons_list_index = [self.indexer.index_of(i) for i in list(cons)]
            self.characters.append(cons_list_index)
            self.labels.append(0)
        for vowel in self.vowel_data:
            vowel_list_index = [self.indexer.index_of(i) for i in list(vowel)]
            self.characters.append(vowel_list_index)
            self.labels.append(1)
        self.characters = torch.tensor(self.characters)
        self.labels = torch.tensor(self.labels)

    def __getitem__(self, idx):
        return self.characters[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.characters)


def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_cons_exs: list of strings followed by consonants
    :param train_vowel_exs: list of strings followed by vowels
    :param dev_cons_exs: list of strings followed by consonants
    :param dev_vowel_exs: list of strings followed by vowels
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNClassifier instance trained on the given data
    """
    train_data = ProcessData(train_cons_exs, train_vowel_exs, vocab_index)
    train_data_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    rnn_classifier = RNNClassifier(vocab_index)
    learning_rate = 0.001
    optimizer = optim.Adam(rnn_classifier.model.parameters(), lr = learning_rate)

    time_s=time.time()
    epochs = 20
    for epoch in range(epochs):
        total_loss = 0.0
        num_epoches = 0
        for char, label in train_data_loader:
            num_epoches+=1
            rnn_classifier.model.train()
            optimizer.zero_grad()
            logits = rnn_classifier.model.forward(char)
            loss = rnn_classifier.loss(logits, label)
            total_loss += loss
            loss.backward()
            optimizer.step()

        total_loss/=num_epoches
        print("Total loss on epoch %i: %f" % (epoch, total_loss))

    print("training time:", time.time()-time_s)
    return rnn_classifier
    

#####################
# MODELS FOR PART 2 #
#####################

class RNNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim =32):
        super(RNNLM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=1, batch_first= True)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x, h_0=None):
        embeddings = self.embeddings(x)
        if h_0 is not None:
            output, h_n = self.gru(embeddings, h_0)
        else:
            output, h_n = self.gru(embeddings) 
        h_n = h_n.squeeze()
        return self.linear(output), self.linear(h_n)

class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e
        :param context: a single character to score
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")

    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0 / self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0 / self.voc_size) * len(next_chars)


class RNNLanguageModel(LanguageModel):
    def __init__(self, vocab_index):
        self.indexer = vocab_index
        self.vocab_size = len(self.indexer)
        self.model = RNNLM(self.vocab_size)
        self.loss = nn.CrossEntropyLoss()
        self.softmax = nn.LogSoftmax()

    def get_next_char_log_probs(self, context):
        context_idx = [self.indexer.index_of(i) for i in context]
        contest_tensor = torch.tensor([context_idx])
        h_n = self.model(contest_tensor)[1].squeeze()
        log_probs = self.softmax(h_n)
        log_probs = log_probs.detach().numpy()
        return log_probs

    def get_log_prob_sequence(self, next_chars, context):
        context_idx = [self.indexer.index_of(i) for i in context]
        next_chars_idx = [self.indexer.index_of(c) for c in next_chars]
        combine_idx = context_idx + next_chars_idx
        combine_idx_tensor = torch.tensor([combine_idx])
        output = self.model(combine_idx_tensor)[0].squeeze()

        log_prob_total = 0
        for i, char_idx in enumerate(next_chars_idx):
            out = output[len(context) -1 + i, :]
            log_prob = self.softmax(out)
            log_prob = log_prob.squeeze()[char_idx]
            log_prob_total += log_prob

        log_probs = log_prob_total.detach().numpy().item()
        return log_probs

class ProcessDataLM(torch.utils.data.Dataset):
    def __init__(self, train_data, indexer, chunk_size):
        super(ProcessDataLM, self).__init__()
        self.data = list(train_data)
        self.indexer = indexer
        self.chunk_size = chunk_size
        self.preprocess_data()

    def preprocess_data(self):
        self.characters = []
        self.labels = []
        self.chunk_chars = [self.data[i: i + self.chunk_size] for i in range(0, len(self.data), self.chunk_size)]
        space_idx = self.indexer.index_of(' ')
        for chunk in self.chunk_chars:
            while len(chunk)!=self.chunk_size:
                chunk.append(' ')
            chunk_index = [self.indexer.index_of(i) for i in chunk]
            self.characters.append([space_idx] + chunk_index[:-1])
            self.labels.append(chunk_index)
            
        self.characters = torch.tensor(self.characters)
        self.labels = torch.tensor(self.labels)

    def __getitem__(self, index):
        return self.characters[index], self.labels[index]
    
    def __len__(self):
        return len(self.characters)


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    """
    train_data = ProcessDataLM(train_text, vocab_index, chunk_size = 50)
    train_loader = DataLoader(train_data, batch_size= 16, shuffle=True)

    rnn_lm = RNNLanguageModel(vocab_index)
    learning_rate = 0.001
    optimizer = optim.Adam(rnn_lm.model.parameters(), lr = learning_rate)

    time_s = time.time()
    epochs = 20
    for epoch in range(epochs):
        total_loss = 0.0
        num_epoches = 0
        for batch_in, batch_out in train_loader:
            num_epoches+=1
            rnn_lm.model.train()
            optimizer.zero_grad()

            logits,h_n = rnn_lm.model.forward(batch_in) 
            logits = logits.view( 16 * 50, len(vocab_index))
            batch_out = batch_out.view(16 * 50)
           
            loss = rnn_lm.loss(logits, batch_out)
            total_loss += loss
            loss.backward()
            optimizer.step()

        total_loss/=num_epoches
        print("Total loss on epoch %i: %f" % (epoch, total_loss))

    print("training time:", time.time()-time_s)
    return rnn_lm
