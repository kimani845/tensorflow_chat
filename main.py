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

# Function to separate words from the sentences given as inputs
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence) # Tokenizing the sentence
    sentence_words = [lemmatizer.lemmatize(word.lower()) 
                        for word in sentence_words] # Lemmatizing the words in the sentence
    return sentence_words # Returning the list of words in the sentence

#  Appending 1 to the index of the words that are present in the sentence and 0 to the rest of the words
def bagw(sentence, words):
    # Seprating the words from the sentence
    sentence_words = clean_up_sentence(sentence) # Getting the list of words in the sentence
    bag = [0] * len(words) # Creating an empty list of 0's with the same length as the words list
    for w in sentence_words: # Looping through the words in the sentence
        for i, word in enumerate(words): # Looping through the words list
            if word == w: # If the word is present in the sentence
                bag[i] = 1 # Setting the index of the word to 1 in the bag list
                
    return np.array(bag) # Returning the bag list as a numpy array