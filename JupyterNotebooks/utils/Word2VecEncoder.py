import numpy as np
import pandas as pd
import nltk
import sys

#text preprocessing, remove stopwords, HTML, non-letters, handle upper/lowercases and tokenize
from nltk.corpus import stopwords
import nltk.data
import re
from gensim.models import word2vec

#For plotting
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt

class Word2VecEncoder:
    def __init__(self, csv_data, text_attributes, down_sample = False, sample_size = None):
        #Combine all text data columns into one string and downsampling if specified
        if down_sample == True:
            rnd_index = np.random.permutation(sample_size)
            buffer_text = buffer_text = csv_data[text_attributes].loc[rnd_index].values
        else:
            buffer_text = csv_data[text_attributes].values

        self.corpus_data = self.combineText(buffer_text)

        #Store text attributes and initialize tokenizer punkt
        self.text_attr = text_attributes
        self.token_punkt = nltk.data.load('tokenizers/punkt/english.pickle')
        self.model = None

    def combineText(self, text):
        #Function to combine word sentences in different columns for similar rows
        corpus = None
        if len(text.shape) == 2:
            corpus = [''] * text.shape[0]
            for row in range(text.shape[0]):
                text_input = ''
                for col in range(text.shape[1]):
                    text_input = text_input + str(text[row, col]) + ' '
                corpus[row] = text_input
        if len(text.shape) == 1:
            corpus = ''
            text_input = ''
            for col in range(len(text)):
                text_input = text_input + str(text[col]) + ' '
            corpus = text_input
        if len(text.shape) > 2:
            print("Error: Does not fit dimensions")
            print("Dimesions - ", len(text.shape))

        return corpus

    def processText(self, corpus = None, tokenizer = None, remove_stops = False):
        raw_text = tokenizer.tokenize(corpus.strip())
        processed_text = []
        for text in raw_text:
            if len(raw_text) > 0:
                text = re.sub("[^a-zA-Z]", " ", text)
                text = text.lower().split()
                if remove_stops == True:
                    stops = set(stopwords.words("english"))
                    text = [w for w in text if not w in stops]
                    processed_text.append(text)
            return processed_text

    def encode(self, num_features = 300, min_word_count = 40, num_workers = 4, replace = True,
                     context = 10, downsampling = 0.01, rm_stops = True):
        #Now to vectorize text for training
        text = []

        print("Processing training text")
        for num in range(len(self.corpus_data)):
            text += self.processText(str(self.corpus_data[num]), self.token_punkt, remove_stops = rm_stops)

        print("Training word2vec model.....")
        self.model = word2vec.Word2Vec(text, workers = num_workers, size = num_features, min_count = min_word_count, window = context, sample = downsampling)
        self.model.init_sims(replace = replace)
        self.vocab = list(self.model.wv.vocab.keys())

    def saver(self, model_name = "w2v model"):
        self.model.save(model_name)

    def plotWords(self, fig_size = (20, 12)):
        #Function to plot out words using matplotlib clustering about a word if stated
        #given some scoring threshold
        X = self.model[self.vocab]

        tsne = TSNE(n_components=2)
        X_tsne = tsne.fit_transform(X)

        df = pd.concat([pd.DataFrame(X_tsne),pd.Series(self.vocab)], axis=1)

        df.columns = ['x', 'y', 'word']

        word_plot = plt.figure(figsize = fig_size)
        ax = word_plot.add_subplot(1,1,1)
        ax.scatter(df['x'], df['y'])

        for i, txt in enumerate(df['word']):
            ax.annotate(txt, (df['x'].iloc[i], df['y'].iloc[i]))

        return ax, word_plot

    def load(self, model_name):
        #Loads model back into class
        try:
            self.model = word2vec.Word2Vec.load(model_name)
            self.vocab = list(self.model.wv.vocab.keys())
        except:
            print("Error: Unable to load model")

    def computeSentence(self, sentence):
        #Function to return the corresponding feature vector for
        #a sentence after processing

        processed_sentence = self.processText(sentence, self.token_punkt, remove_stops = True)
        compat_text = []

        for word in processed_sentence:
            if word in self.vocab:
                compat_text += word

        return compat_text
