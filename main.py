# This main.py file will be used to run the model 
import random 
import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# initializing the lemmatizer
lemmatizer = WordNetLemmatizer()
nltk.download('punkt')

# loading the files that were saved in the training.py file
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')