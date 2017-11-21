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

        self.corpus_data = [''] * buffer_text.shape[0]
        for row in range(buffer_text.shape[0]):
            text_input = ''
            for col in range(buffer_text.shape[1]):
                text_input = text_input + str(buffer_text[row, col]) + ' '
            self.corpus_data[row] = text_input

        #Store text attributes and initialize tokenizer punkt
        self.text_attr = text_attributes
        self.token_punkt = nltk.data.load('tokenizers/punkt/english.pickle')

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
                     context = 10, downsampling = 0, rm_stops = True):
        #Now to vectorize text for training
        text = []

        print("Processing training text")
        for num in range(len(self.corpus_data)):
            text += self.processText(str(self.corpus_data[num]), self.token_punkt, remove_stops = rm_stops)

        print("Training word2vec model.....")
        self.model = word2vec.Word2Vec(text, workers = num_workers, size = num_features, min_count = min_word_count, window = context, sample = downsampling)
        self.model.init_sims(replace = replace)

    def saver(self, model_name = "w2v model"):
        self.model.save(model_name)

    def plotWords(self, fig_size = (20, 12)):
        #Function to plot out words using matplotlib clustering about a word if stated
        #given some scoring threshold
        vocab = list(self.model.wv.vocab)
        X = self.model[vocab]

        tsne = TSNE(n_components=2)
        X_tsne = tsne.fit_transform(X)

        df = pd.concat([pd.DataFrame(X_tsne),pd.Series(vocab)], axis=1)

        df.columns = ['x', 'y', 'word']

        word_plot = plt.figure(figsize = fig_size)
        ax = word_plot.add_subplot(1,1,1)
        ax.scatter(df['x'], df['y'])

        for i, txt in enumerate(df['word']):
            ax.annotate(txt, (df['x'].iloc[i], df['y'].iloc[i]))

        return ax, word_plot
