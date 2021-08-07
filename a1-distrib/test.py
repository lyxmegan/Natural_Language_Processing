import re
from utils import *
from collections import Counter
import nltk
from nltk.corpus import stopwords 
def split_sentence(sentence): 
    string = re.sub(r"[^A-Za-z0-9']", " ", sentence)
    string = re.sub(r"''", '', string)
    result = string.split()
    stop_words = set(stopwords.words('english')) 
    result = [w for w in result if not w in stop_words]  
    print(result)
    # indices = []
    # indexer = Indexer()
    # for word in set(result): 
    #     index = Indexer.index_of(indexer, word)
    #     indices.append(index)
    # print(indices)
    return result




sentence = "-LRB- `` take ourselves Is Take is "
result = split_sentence(sentence)
print(result)

def add_features(sentence):    
    stop = []
    indexer = Indexer()
    for word in sentence: 
        word = word.lower()
        if word not in stop: 
            indexer.add_and_get_index(word)

def extract_features(sentence, add_to_indexer: bool= False) -> Counter: 
    indexer = Indexer()
    if add_to_indexer: 
        add_features(sentence)
    count = Counter()
    for word in sentence: 
        word = word.lower()
        if indexer.contains(word): 
            index = indexer.index_of(word)
            count.update([index]) 
        
        return count

count = extract_features(result)
print(count)