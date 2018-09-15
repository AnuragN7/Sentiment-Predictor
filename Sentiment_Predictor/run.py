import json
import nltk 
import Sentiment_Predictor

data = [] # Will store the contents of the JSON file

with open('sentiment_analysis_data.json', 'r') as file_name:
	data = json.load(file_name) # Load the json file

data_length = len(data)
trimmed_data = data[:int(data_length/2)] # Only use a portion of the data to train the predictor

tweets = [tweet_info['text'] for tweet_info in trimmed_data] # Extract the actual tweet text from the data
labels = [tweet_info['label'] for tweet_info in trimmed_data] # Exract every label from each dictionary
word_data = [nltk.word_tokenize(text) for text in tweets] # Having words in lists makes them easier to work with

feature_data = list(zip(word_data, labels)) # The feature data that will be fed in to the sentiment predictor



