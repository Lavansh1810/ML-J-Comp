import json
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import random
import datetime
import webbrowser
import pyttsx3
# from pygame import mixer
# import speech_recognition as sr


import colorama
colorama.init()
from colorama import Fore, Style, Back

import random
import pickle
#speech
from talk import take_command

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
volume = engine.getProperty('volume')
engine.setProperty('volume', 10.0)
rate = engine.getProperty('rate')

engine.setProperty('rate', rate - 25)
with open('intents.json') as file:
    data = json.load(file)

def chat():
    #load trained model
    model = keras.models.load_model('chat-model')

    #load tokenizer object
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    #load label encoder object
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    #parameters
    max_len = 20
    while True:
        print(Fore.LIGHTBLUE_EX + 'User: ' + Style.RESET_ALL, end = "")
        inp = input()
        if inp.lower() == 'quit':
            print(Fore.GREEN + 'Bot:' + Style.RESET_ALL, "Take care. See you soon.")
            break
        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]), truncating = 'post', maxlen = max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])
        for i in data['intents']:
            if i['tag'] == tag:
                print(Fore.GREEN + 'Bot:' + Style.RESET_ALL, np.random.choice(i['responses']))

    





print("Hi there! Can you tell if you would like to chat with me or talk to me ?")
val=input()

if val.lower()=='chat':
    chat()
elif val.lower()=='talk':
    #load trained model
    model = keras.models.load_model('chat-model')

    #load tokenizer object
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    #load label encoder object
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    #parameters
    max_len = 20
    while True:
        print(Fore.LIGHTBLUE_EX + 'Listening: ' + Style.RESET_ALL, end = "")
        inp = take_command()
        if inp.lower() == 'quit':
            print(Fore.GREEN + 'Cutie Bot:' + Style.RESET_ALL, "Take care. See you soon.")
            break
        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]), truncating = 'post', maxlen = max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])
        for i in data['intents']:
            if i['tag'] == tag:
                print(Fore.GREEN + 'Pandora:' + Style.RESET_ALL, np.random.choice(i['responses']))