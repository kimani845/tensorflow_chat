# importing the reqired libraries
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import  nltk 

from keras.models import Sequential
from ntlk.tokenize import word_tokenize
from keras.layers import Dense, Activation, Dropout , Embedding, LSTM, Bidirectional
from keras.optimizers import Adam
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('wordnet')

# Load the intents file
intents = json.loads(open('intents.json').read())
#  Creating empty lists to hold the data
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',', "'s", "'m", "'ll", "'re", "'ve", "'d", "'t", "n't", "(", ")", "[", "]", "{", "}", ";", ":"]

# Loop through the intents and extract the patterns and responses
for intent in intents['intents']:
    for pattern in intent['patterns']:
        
        # Tokenize each word in the pattern
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list) # Add the words to the words list
        
        # Add the pattern and intent to the documents list
        documents.append((word_list, intent['tag']))
        
        # Adding intent to the classes list if it's not already there
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
    
# Storing the root woeds or lema
words = [lemmatizer.lemmatize(w.lower())
            for w in words if w not in ignore_words]
words = sorted(set(words)) # Remove duplicates and sort the words

#  Saving the words and classes to binary files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# classifying data into 0's and 1's for the neural network to work with
# Creating an empply list to hold the training data
training = []
output_empty = [0] * len(classes) # Creating an empty list of 0's with the same length as the classes
for document in documents:
    bag = [] # Creating an empty list to hold the bag of words
    word_patterns = r'\b' + r'\w+\b' # Regular expression to match words 
    word_list = document[0] # Getting the word list from the document
    word_list = [lemmatizer.lemmatize(
        w.lower()) for w in word_list] # Lemmatizing the words in the word list 
    for word in words:
        bag.append(1) if word in word_list else bag.append(0)
        
    
    # Making a copy of the output_empty 
    output_row = list(output_empty) # Creating a copy of the output_empty list
    output_empty[classes.index(document[1])] = 1 # Setting the index of the class to 1 in the output_row list
    training.append([bag, output_row]) # Appending the bag and output_row to the training list
    

    