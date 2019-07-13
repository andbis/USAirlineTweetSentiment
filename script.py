import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
import numpy as np
from library import pre_processing, algo_run, TfidfEmbeddingVectorizer, MeanEmbeddingVectorizer, neural

if __name__ == "__main__" and len(sys.argv) == 1:
	print('\nSyntax: python script.py algorithm feature_set lemma(optional)\n')
	print('Available algorithms: "log" for LogisticRegression, "svm" for Support Vector Machine, "nb" for Multinomial Naive Bayes(for feature_sets 1-3) + Gaussian Naive Bayes(for feature_sets 4-6), "neural" for network with conv1d layers \n')
	print('Lemmatization is by default disabled, to enable write "lemma" after feature_set\n')
	print('Available feature_sets: "1" for unigram, "2" for bigram, "3" for combined unigram and bigram feature_sets, "4" Word Embedding Self 200d, "5" Glove Twitter Pre-trained Embedding 27b 200d, "6" Additional pre-trained Twitter Embedding 400 mill tweets 400d\n Examples:')
	print('python script.py log 2  - for LogisticRegression model with bigram feature')
	print('python script.py nb 13  - for Multinomial Naive Bayes model to run with both unigram and combined feature')
	print('python script.py svm 123  - for Support Vector Machine model to run with first 3 feature_sets\n')

	
else:
	data = pd.read_csv('data/Tweets-airline-sentiment.csv')
	text = data.loc[:,'text']
	labels = data.loc[:,'airline_sentiment']
	logs = ['Unigram', 'Bigram', 'Combined uni-bigram', 'Self Embedding', 'Glove Embedding', 'Twitter Embedding']
	if sys.argv[1] == 'hist':
		classes = ['negative', 'neutral', 'positive']
		y = [classes.index(a) for a in labels]
		airline = data.loc[:,'airline']
		crosstab = pd.crosstab(airline,labels).apply(lambda r: r/r.sum(), axis=1)
		ax = crosstab.plot(kind='bar', stacked=False, colormap="tab20c",width=0.8)
		ax.set_xlabel("Airline")
		ax.set_ylabel("Tweet amount in percent")
		ax.legend(title="Sentiment")
		plt.savefig('histofclasses.png')
		print('histofclasses.png saved to current directory')
		neg_count = y.count(0)
		y_len = len(y)
		print('Number of negative examples: %d, Total Examples: %d, Baseline Acc: %f' % (neg_count, y_len, neg_count/y_len))

	elif len(sys.argv) > 1:
		try:
			if sys.argv[3] == 'lemma':
				lemmatize = True
		except IndexError:
			lemmatize = False

		algos = ['log', 'svm', 'nb', 'neural']
		if sys.argv[1] not in algos:
			raise TypeError('Input algorithm: %s is not part of program, Please follow the suggested syntax: python script.py algorithm feature_set lemma(optional)\n\
				"python script.py" for additional information' % sys.argv[1])

		try:
			number = [a for a in sys.argv[2]]
			c_algo = algos.index(sys.argv[1])
			if c_algo != 3:
				while (len(number) != 0):
					try:
						c_run = int(number.pop(0))
					except ValueError:
						raise ValueError('Please write a number between 1-6 for choice of feature_set: 1: unigram, 2: bigram, 3: combined, 4: Self-embedding, 5: glove embedding, 6: twitter embedding')
					print('Starting pre-processing of %s-model' % logs[c_run-1])
					run_data = pre_processing(c_run, text, labels, lemma=lemmatize)
					algo_run(c_algo, c_run, run_data[0], run_data[1], run_data[2], run_data[3])
			else:
				#neural
				while (len(number) != 0):
					try:
						c_run = int(number.pop(0))
					except ValueError:
						raise ValueError('Please write a number between 1-6 for choice of feature_set: 1: unigram, 2: bigram, 3: combined, 4: Self-embedding, 5: glove embedding, 6: twitter embedding')
					if c_run <= 3:
						raise TypeError('Only embeddings are available for Convolutional Network:\n pyhon script.py neural 456')
					print('Preparing pre-processing of %s-model' % logs[c_run-1])
					w2v = pre_processing(c_run, text, labels, lemma=lemmatize, not_neural=False)
					neural(w2v, text, labels)

		except IndexError:
			print('Please follow the suggested syntax: python script.py algorithm feature_set lemma(optional)\n\
				"python script.py" for additional information')
