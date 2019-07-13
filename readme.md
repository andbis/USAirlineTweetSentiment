<h1>Readme file for simple program to run sentiment analysis on US Airline Twitter data set</h1>

Logistic Regression, Support Vector Machine, Naive Bayes and simple convolutional network used

Included in the rep should be the following:
- script.py - run script from where the algorithms are executed
- library.py - utilities used by the script
- data/ - location to move data resources to 
- word2vec_twitter_master/ - folder containing script and utilities to unpack download embedding model

<h2>Requirements</h2>

Program and code is developed and tested on OSX Version 10.13.4 Anaconda built python 3.6 environment
The following packages are required to successfully execute the code:
- matplotlib
	.pyplot
- pandas
- numpy
- gensim
- sklearn
	.pipeline
	.metrics
	.feature_extraction
	.model_selection
	.linear_model
	.naive_bayes
	.svm
- keras
	.utils
	.layers
	.preprocessing
	.models
- spacy ("en_core_web_sm" model used)

Move "Tweets-ariline-sentiment.csv" to "data" folder

Embedded models, to make the glove and twitter embedding model work one must download and locate in the "data" folder: 

GloVe Twitter 27b 200d pre-trained model (glove.twitter.27B.200d.txt): http://nlp.stanford.edu/data/glove.twitter.27B.zip

Fredéric Godin Twitter 400mill 400d pre-trained model (word2vec_twitter_model.bin): https://www.fredericgodin.com/software/ - alternatively https://drive.google.com/file/d/10B7cvx3xN7Ef_FxwIO8sigd1J1Ibe6Lu/view?usp=sharing

<h2>Execution of the program</h2>
In the terminal navigate to the unpacked "sentimentanalysis" folder. 
Use the following syntax for execution **"python script.py algorithm feature-set lemmatization(optional)"** eg. "python script.py log 2" for Logistic regression model with bigram feature set. Lemmatization is optional, i.e. if not included no lemmatization will be made. 

Available algorithms are: 
- "log" for LogisticRegression
- "svm" for Support Vector Machine
- "nb" for Multinomial Naive Bayes(for feature_sets 1-3) + Gaussian Naive Bayes(for feature_sets 4-6) 
- "neural" for network with conv1d layers ('only works with feature sets 4-6')

Available feature sets are:
- "1" for unigram
- "2" for bigram
- "3" for combined uni+bigram
- "4" for self embedding
- "5" for GloVe Twitter embedding
- "6" for Frederic Godin Twitter embedding