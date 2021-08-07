# tokenizer.py

import re

#understanding: 
#result = re.sub(pattern, repl, string, count=0, flags=0);
#result = re.sub('abc',  '',    input)           # Delete pattern abc
#result = re.sub('abc',  'def', input)           # Replace pattern abc -> def
#result = re.sub(r'\s+', ' ',   input)           # Eliminate duplicate whitespaces using wildcards
#result = re.sub('abc(def)ghi', r'\1', input)    # Replace a string with a part of itself
# Tokenizes a string. Takes a string (a sentence), splits out punctuation and contractions, and returns a list of
# strings, with each string being a token.
def tokenize(string):
    # print(repr(string))
    string = re.sub(r"[^A-Za-z0-9(),.!?\'`\-\"]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\-", " - ", string)
    string = re.sub(r"\"", " \" ", string)
    # We may have introduced double spaces, so collapse these down
    string = re.sub(r"\s{2,}", " ", string)
    return list(filter(lambda x: len(x) > 0, string.split(" ")))

if __name__=="__main__":
    print(repr(tokenize("said.")))
    print(repr(tokenize("said?\"")))
    print(repr(tokenize("I didn't want to, but I said \"yes!\" anyway.")))