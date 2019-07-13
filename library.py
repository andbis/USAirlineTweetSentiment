from collections import defaultdict
import gensim
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath("word2vec_twitter_master/"))
import word2vecReader
import pandas as pd
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import LinearSVC

def pre_processing(c, text, labels, max_features=None, lemma=False, not_neural=True):
    if lemma:
        text = lemmatization(text)

    if c == 1:
        #unigram feature
        print('Vectorizing unigram')
        vectorizer_uni = CountVectorizer(ngram_range=(1,1), binary=True, max_features=max_features).fit(text)
        X = vectorizer_uni.transform(text).toarray()

    elif c == 2: 
        #bigram feature 
        print('Vectorizing bigram')
        vectorizer_bi = CountVectorizer(ngram_range=(2,2), binary=True, max_features=max_features).fit(text)
        X = vectorizer_bi.transform(text).toarray()
    
    elif c == 3:
        #combined unigram + bigram array
        print('Combining')
        vectorizer_uni = CountVectorizer(ngram_range=(1,1), binary=True, max_features=max_features).fit(text)
        vectorizer_bi = CountVectorizer(ngram_range=(2,2), binary=True, max_features=max_features).fit(text)
        X = FeatureUnion([("unigram", vectorizer_uni), ("bigram", vectorizer_bi)]).transform(text).toarray()

    elif c == 4:
        #self training - embedding
        print('Embedding data with self_embedding')
        sentences = [a for a in text]
        model = gensim.models.Word2Vec(sentences, size=200,iter=30)
        w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}
        if not_neural == False:
            return w2v
        print('Vectorizing text')
        vectorizer = TfidfEmbeddingVectorizer(w2v)
        vectorizer.fit(text, labels)
        X = vectorizer.transform(text)

    elif c == 5:
        #pre-trained word embedding GloVe twitter 200d
        #https://nlp.stanford.edu/projects/glove/
        print('Embedding data with pre-trained "glove.twitter.27B.200d", this can take some time...')
        with open("data/glove.twitter.27B.200d.txt", "r") as lines:
            w2v = {line.split()[0]: np.array(line.split()[1:], dtype='float32') for line in lines}
        if not_neural == False:
            return w2v
        print('Vectorizing text')
        vectorizer = TfidfEmbeddingVectorizer(w2v)
        vectorizer.fit(text, labels)
        X = vectorizer.transform(text)

    elif c == 6:
        #https://www.fredericgodin.com/software/
        #Pre-traned word embedding twitter data
        print('Embedding data with pre-trained 400mill tweets')
        embed_space = word2vecReader.twitter_embedding() #word2vecreader is from: https://github.com/loretoparisi/word2vec-twitter
        w2v = {w: vec for w, vec in zip(embed_space.vocab, embed_space.syn0)}
        if not_neural == False:
            return w2v
        print('Vectorizing text')
        vectorizer = TfidfEmbeddingVectorizer(w2v)
        vectorizer.fit(text, labels)
        X = vectorizer.transform(text)
    else: raise TypeError('Wrong key input for feature_set choice: %d; 1: unigram, 2: bigram, 3: combined, 4: Self-embedding, 5: glove embedding, 6: twitter embedding')

    #split test train
    print('Splitting train_test_split')
    xtrain, xtest, ytrain, ytest = train_test_split(X, labels, test_size=0.2, random_state=442)
    run_data = [xtrain, xtest, ytrain, ytest]
    print('Run data has %d dimensions' % xtrain.shape[1])
    return run_data

def algo_run(a, c, xtrain, xtest, ytrain, ytest):
    feature_sets = ['Unigram', 'Bigram', 'Combined uni-bigram', 'Self-embedding', 'Glove-twitter-embedding', 'twitter_embedding']
    models = [LogisticRegression, LinearSVC, MultinomialNB]     
    model = models[a]
    c_feature_set = feature_sets[c-1]
    if a == 2 and c >= 4:
        print('Fitting Gaussian Naive Bayes for word-embedding')
        model = GaussianNB

    print('Starting %s fit of %s feature' % (model, c_feature_set))
    algorithm = model().fit(xtrain, ytrain)
    print('Accuracy:', algorithm.score(xtest, ytest))
    predicted = algorithm.predict(xtest)
    print('Classification report:\n',classification_report(ytest, predicted))
    print('Confusion matrix:\n', confusion_matrix(ytest, predicted))
    return  

def neural(w2v, text, labels):
    first = list(w2v.keys())[0]
    MAX_SEQUENCE_LENGTH = 150
    VALIDATION_SPLIT = 0.2
    EMBEDDING_DIM = len(w2v[first])

    from keras import utils
    from keras.layers import Conv1D, MaxPooling1D, Embedding, Dense, Input, GlobalMaxPooling1D, Flatten
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.models import Model
    
    classes = ['negative', 'neutral', 'positive']
    labels = [classes.index(a) for a in labels]
    texts = [a for a in text]
    tokenizer = Tokenizer()#num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)


    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    labels = utils.to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # split the data into a training set and a validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]

    
    #Preparing embedding
    #At this point we can leverage our embedding_index dictionary and our word_index to compute our embedding matrix:
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    embedding_vector = np.zeros((EMBEDDING_DIM))
    good = 0
    for word, i in word_index.items():
        try:
            embedding_vector = w2v[word]
        except KeyError:
            pass
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
            good += 1
    prototype = np.zeros((EMBEDDING_DIM))
    co = sum([1 if (prototype == a).all() else 0 for a in embedding_matrix]) - 1
    print('number of zero vectors:', co)
    print('Found word vectors for %d words of a total of %d unique words' % (good, len(word_index)))

    #We load this embedding matrix into an Embedding layer. Note that we set trainable=False to prevent the weights from being updated during training.
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(len(classes), activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])



    model.fit(x_train, y_train, validation_data=(x_val, y_val),
             epochs=10, batch_size=128)

    print('Done')

    return

def lemmatization(text):
    print('Starting lemmatization of text')
    import spacy
    nlp = spacy.load("en_core_web_sm")
    text = pd.Series(text).str.lower()
    # replace airline company twitter names
    text = pd.Series(text).str.replace(r'@\w+', '')
    text = pd.Series(text).str.replace('@[^\s]+','')
    text = pd.Series(text).str.replace(r'http.?://[^\s]+[\s]?', '')
    text = pd.Series(text).str.replace(r'&amp', '')
    text = pd.Series(text).str.replace(r'&gt', '')
    text = pd.Series(text).str.replace(r'&lt', '')
    text = pd.Series(text).str.replace('[^\w\s]','')
    text = pd.Series(text).str.lstrip()
    text = pd.Series(text).str.rstrip()
    # common spelling mistakes
    text = pd.Series(text).str.replace(r'\bcudtomers\b', 'customers')
    text = pd.Series(text).str.replace(r'\bppl\b', 'people')
    text = pd.Series(text).str.replace(r'\biphone\b', 'phone')
    text = pd.Series(text).str.replace(r'#([^\s]+)', r'\1')
    text = text.apply(lambda row: [w.lemma_ for w in nlp(row)])
    text = [' '.join(i) for i in text]
    return text


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec, dim=0):
        self.word2vec = word2vec
        self.word2weight = None
        if len(word2vec)>0:
            first = list(word2vec.keys())[0]
            self.dim=len(word2vec[first])
        
        
    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf, 
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self
    
    def transform(self, X):
        return np.array([np.mean([self.word2vec[w] * self.word2weight[w] \
            for w in words if w in self.word2vec] \
            or [np.zeros(self.dim)], axis=0) for words in X])

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        first = list(word2vec.keys())[0]
        self.dim=len(word2vec[first])

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
