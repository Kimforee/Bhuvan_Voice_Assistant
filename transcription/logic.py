# transcription_app/logic.py
import os
import torch
import nltk
import numpy as np
import torch.nn as nn
from time import sleep
from queue import Queue
from sys import platform
import speech_recognition as sr
from unittest.mock import sentinel
from django.http import JsonResponse
from datetime import datetime, timedelta
from nltk.stem.porter import PorterStemmer

Stemmer = PorterStemmer()
global transcription_text

def initialize_recorder(sample_rate=16000, energy_threshold=1000, default_microphone=None):
    recorder = sr.Recognizer()
    recorder.energy_threshold = energy_threshold
    recorder.dynamic_energy_threshold = False
    print("initializing recorder")
    if 'linux' in platform:
        mic_name = default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return None
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=sample_rate, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=sample_rate)

    with source:
        recorder.adjust_for_ambient_noise(source)

    return recorder, source

# Neural network
class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(NeuralNet,self).__init__()
        self.l1 = nn.Linear(input_size,hidden_size)
        self.l2 = nn.Linear(hidden_size,hidden_size)
        self.l3 = nn.Linear(hidden_size,num_classes)
        self.relu = nn.ReLU()

    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out
 
# ----------------------------------------------------------------
# sentence processing

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

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