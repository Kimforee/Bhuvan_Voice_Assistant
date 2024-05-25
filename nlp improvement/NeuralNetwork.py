from unittest.mock import sentinel
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

Stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def tokenize(sentence):
    tokens = nltk.word_tokenize(text)
    
    # Remove stop words
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def stem(word):
    print(word)
    word=''.join(word)
    print(word)
    return Stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence,words):
    sentence_word = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words),dtype = np.float32)

    for idx , w in enumerate (words):
        if w in sentence_word:
           bag[idx] = 1

    return bag

import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.l1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.l2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.l3(out)
        return out