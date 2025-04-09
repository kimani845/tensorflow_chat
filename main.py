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
    # Seprating the words from the input sentence
    sentence_words = clean_up_sentence(sentence) # Getting the list of words in the sentence
    bag = [0] * len(words) # Creating an empty list of 0's with the same length as the words list
    for w in sentence_words: # Looping through the words in the sentence
        for i, word in enumerate(words): # Looping through the words list
            if word == w: # If the word is present in the sentence
                bag[i] = 1 # Setting the index of the word to 1 in the bag list
                
    return np.array(bag) # Returning the bag list as a numpy array

# A function to predict the class of the input sentence
def predict_class(sentence, model):
    # Getting the bag of words for the input sentence
    p = bagw(sentence, words) # Getting the bag of words for the input sentence
    res = model.predict(np.array([p]))[0] # Predicting the class of the input sentence
    ERROR_THRESHOLD = 0.25 # Setting the error threshold
    results = [[i, r] for i, r in enumerate(res) 
                if r > ERROR_THRESHOLD] # Getting the index and probability of the predicted class
    results.sort(key=lambda x: x[1], reverse=True) # Sorting the results in descending order of probability
    return_list = [] # Creating an empty list to hold the predicted classes and probabilities
    for r in results: # Looping through the results
        return_list.append({'intent': classes[r[0]],
                            'probability': str(r[1])}) # Appending the predicted class and probability to the return_list
    
    return return_list # Returning list of predicted classes and probabilities

# A function to get the response for the input sentence
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent'] # Getting the predicted class
    list_of_intents = intents_json['intents'] # Getting the list of intents from the intents.json file
    for i in list_of_intents: # Looping through the list of intents
        if i['tag'] == tag: # If the predicted class matches the intent tag
            response = random.choice(i['responses']) # Getting a random response from the intent
            break
    
    return response # Returning the response