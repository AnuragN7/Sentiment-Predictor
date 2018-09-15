import nltk
from nltk.corpus import movie_reviews 
# from nltk.corpus import stopwords


class Sentiment_Predictor:
	""" This class is used to predict the sentiment of a tweet. Tweet's from Donald Trump's 
		Twitter handle will be used in this program
	"""
	def __init__(self, document):
		""" Instantiates a sentiment_predictor object 

		Args:
		training_data: Contains the tweets and labels

		"""
		self.classifier = self.get_classifier(document)

	def get_classifier(self, document):
		""" Returns a classifier that will be used to predict the 
			sentiment of a tweet
		Args:
		document: contains the tweets and labels

		"""

		feature_sets = [(self.get_features(tweet), label) for (tweet, label) in document] # Get the featuresets for every single tweet
		classifier = nltk.NaiveBayesClassifier.train(feature_sets) # Train a Naive Bayes classifer
		return classifier # Return the classifier

	def get_features(self, tweet):
		""" Extracts certain features for a tweet
		
		Args: 
		
		tweet: A basic twitter tweet

		"""
		movie_review_words = movie_reviews.words()
		all_words = nltk.FreqDist(w.lower() for w in movie_review_words)
		word_features = list(all_words.keys())[:1000] # Data to be for the feature extractor
	
		feature_set = dict()
		for word in word_features:
			feature_set['contains(%s)' % word] = (word in tweet)
		return feature_set

	def predict_sentiment(self, tweet):
		""" As the name suggest, this method simply predicts the sentiment of a tweet

		Args: 

		tweet: A tweet whose sentiment needs to be predicted
		"""

		return self.classifier.classify(tweet)
