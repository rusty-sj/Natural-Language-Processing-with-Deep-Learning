import string
from collections import Counter, OrderedDict

import matplotlib.pyplot as plt
import nltk
import numpy as np

nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer


class UnimplementedFunctionError(Exception):
    pass


class Vocabulary:

    def __init__(self, corpus):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.word2idx, self.idx2word, self.freq = self.build_vocab(corpus)
        self.size = len(self.word2idx)

    def most_common(self, k):
        freq = sorted(self.freq.items(), key=lambda x: x[1], reverse=True)
        return [t for t, f in freq[:k]]

    def text2idx(self, text):
        tokens = self.tokenize(text)
        return [self.word2idx[t] if t in self.word2idx.keys() else self.word2idx['UNK'] for t in tokens]

    def idx2text(self, idxs):
        return [self.idx2word[i] if i in self.idx2word.keys() else 'UNK' for i in idxs]

    ###########################
    #  TASK 1.1           	  #
    ###########################
    def tokenize(self, text):
        """

        tokenize takes in a string of text and returns an array of strings splitting the text into discrete tokens.

        :params:
        - text: a string to be tokenize, e.g. "The blue dog jumped, but not high."

        :returns:
        - tokens: a list of strings derived from the text, e.g. ["the", "blue", "dog", "jumped", "but", "not", "high"]
        for word-level tokenization

        """
        # print("Sentence: ", text)
        text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
        text = text.translate(str.maketrans('', '', string.digits))  # remove digits
        text = text.lower()  # convert text to lowercase
        tokens = text.split()  # split on whitespace
        lemmatized_words = [self.lemmatizer.lemmatize(token) for token in tokens]  # lemmatization
        # print("Tokenization: ", lemmatized_words)
        return lemmatized_words

    ###########################
    #  TASK 1.2            	  #
    ###########################
    def build_vocab(self, corpus):
        """

        build_vocab takes in list of strings corresponding to a text corpus, tokenizes the strings, and builds a finite vocabulary

        :params:
        - corpus: a list string to build a vocabulary over

        :returns:
        - word2idx: a dictionary mapping token strings to their numerical index in the dictionary e.g. { "dog": 0, "but":1, ..., "UNK":129}
        - idx2word: the inverse of word2idx mapping an index in the vocabulary to its word e.g. {0: "dog", 1:"but", ..., 129:"UNK"}
        - freq: a dictionary of words and frequency counts over the corpus (including words not in the dictionary), e.g. {"dog":102, "the": 18023, ...}

        """

        word2idx = {}
        idx2word = {}
        idx_counter = 0

        corpus_tokens = []
        for sentence in corpus:
            corpus_tokens.extend(self.tokenize(sentence))

        freq = Counter(corpus_tokens)
        freq = OrderedDict(freq.most_common())
        for key, value in freq.items():
            # if key not in self.stop_words:
            if value > 55 and key not in self.stop_words:
                word2idx[key] = idx_counter
                idx2word[idx_counter] = key
                idx_counter += 1
        word2idx['UNK'] = idx_counter
        idx2word[idx_counter] = 'UNK'
        idx_counter += 1
        # print('word2idx', word2idx)
        # print('word2idx', idx2word)
        # print('freq', freq)
        print("vocab size:", len(word2idx))
        return word2idx, idx2word, freq

    ###########################
    #  TASK 1.3               #
    ###########################
    def make_vocab_charts(self):
        """

        make_vocab_charts plots word frequency and cumulative coverage charts for this vocabulary. See handout for more details


        """

        fig = plt.figure()
        plt.title('Token Frequency Distribution')
        plt.xlabel(xlabel='Token ID (sorted by frequency high to low)')
        plt.ylabel(ylabel='Frequency')
        plt.yscale('log')
        plt.plot(np.arange(len(self.freq)), np.array(list(self.freq.values())), label='word frequencies')
        plt.axhline(y=50, c='red', linestyle='dashed', label="freq=50")
        plt.legend()
        # plt.show()
        plt.savefig('tfd.png')

        training_tokens = sum(self.freq.values())
        cutoff = 0
        occurrences = []
        for word in self.freq.keys():
            occurrences.append(cutoff / training_tokens)
            # if 0.91 < (cutoff / training_tokens) < 0.92:
            #     print(cutoff / training_tokens, self.freq[word], self.word2idx[word])
            cutoff += self.freq[word]
        fig = plt.figure()
        plt.title('Cumulative Fraction Covered')
        plt.xlabel(xlabel='Token ID (sorted by frequency high to low)')
        plt.ylabel(ylabel='Fraction of Token Occurrences Covered')
        plt.plot(np.arange(len(self.freq)), np.array(occurrences), label='cutoff curve')
        plt.axvline(x=6530, c='red', linestyle='dashed', label="y=0.91")
        plt.legend()
        # plt.show()
        plt.savefig('cfc.png')

    def get_wordnet_pos(self, nltk_tag):
        """

         Identify POS of inout word for using in lemmatizer's argument to make it work better
        :param nltk_tag: word string

        :return: tagged POS for the input word string

        """
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(nltk_tag, wordnet.NOUN)
