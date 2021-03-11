import logging
import os.path
import random

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from sklearn.manifold import TSNE
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm

from Vocabulary import Vocabulary

random.seed(42)
np.random.seed(42)

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


class UnimplementedFunctionError(Exception):
    pass


###########################
#  TASK 2.2               #
###########################

def compute_cooccurrence_matrix(corpus, vocab):
    """

        compute_cooccurrence_matrix takes in list of strings corresponding to a text corpus and a vocabulary of size N and returns
        an N x N count matrix as described in the handout. It is up to the student to define the context of a word

        :params:
        - corpus: a list strings corresponding to a text corpus
        - vocab: a Vocabulary object derived from the corpus with N words

        :returns:
        - C: a N x N matrix where the i,j'th entry is the co-occurrence frequency from the corpus between token i and j in the vocabulary

        """

    if os.path.isfile('C.npy') and os.path.isfile('N.npy'):
        return np.load('C.npy'), np.load('N.npy')

    vocab_size, window, N = vocab.size, 5, 0
    C = np.zeros((vocab_size, vocab_size))
    for sentence in corpus:
        sent_idxs = vocab.text2idx(sentence)
        N += len(sent_idxs) - (2 * window)
        for i, idx in enumerate(sent_idxs):
            if window <= i < (len(sent_idxs) - window):
                C[idx][idx] += 1
                set_idxs = {idx}
                for j in range((i - window), (i + window + 1)):
                    if sent_idxs[j] not in set_idxs:
                        C[idx][sent_idxs[j]] += 1
                        set_idxs.add(sent_idxs[j])
    np.save('C.npy', C)
    np.save('N.npy', N)
    return C, N


###########################
# TASK 2.3                #
###########################

def compute_ppmi_matrix(corpus, vocab):
    """

        compute_ppmi_matrix takes in list of strings corresponding to a text corpus and a vocabulary of size N and returns
        an N x N positive pointwise mutual information matrix as described in the handout. Use the compute_cooccurrence_matrix function.

        :params:
        - corpus: a list strings corresponding to a text corpus
        - vocab: a Vocabulary object derived from the corpus with N words

        :returns:
        - PPMI: a N x N matrix where the i,j'th entry is the estimated PPMI from the corpus between token i and j in the vocabulary

        """

    if os.path.isfile('PPMI.npy'):
        return np.load('PPMI.npy')
    C, N = compute_cooccurrence_matrix(corpus, vocab)
    C += 1e-9
    PPMI = np.zeros((vocab.size, vocab.size))
    for i in range(len(C)):
        for j in range(len(C)):
            PPMI[i][j] = max(0, (np.log((C[i][j] * N) / (C[i][i] * C[j][j]))))
    np.save('PPMI.npy', PPMI)
    return PPMI


################################################################################################
# Main Skeleton Code Driver
################################################################################################
def main_freq():
    logging.info("Loading dataset")
    dataset = load_dataset("ag_news")
    dataset_text = [r['text'] for r in dataset['train']]
    dataset_labels = [r['label'] for r in dataset['train']]

    logging.info("Building vocabulary")
    vocab = Vocabulary(dataset_text)
    # vocab = Vocabulary(dataset_text[:5])
    # vocab = Vocabulary(['With Funding From Jeff Bezos,\n MethaneSAT Picks Elon Musk\'s SpaceX for 2022 Launch.',
    #                     'We couldn\'t ask for a more capable launch partner.'])
    vocab.make_vocab_charts()
    plt.close()
    # plt.pause(0.01)

    logging.info("Computing PPMI matrix")
    PPMI = compute_ppmi_matrix([doc['text'] for doc in dataset['train']], vocab)

    logging.info("Performing Truncated SVD to reduce dimensionality")
    word_vectors = dim_reduce(PPMI)

    logging.info("Preparing T-SNE plot")
    plot_word_vectors_tsne(word_vectors, vocab)


def dim_reduce(PPMI, k=16):
    U, Sigma, VT = randomized_svd(PPMI, n_components=k, n_iter=10, random_state=42)
    SqrtSigma = np.sqrt(Sigma)[np.newaxis, :]

    U = U * SqrtSigma
    V = VT.T * SqrtSigma

    word_vectors = np.concatenate((U, V), axis=1)
    word_vectors = word_vectors / np.linalg.norm(word_vectors, axis=1)[:, np.newaxis]

    return word_vectors


def plot_word_vectors_tsne(word_vectors, vocab):
    coords = TSNE(metric="cosine", perplexity=50, random_state=42).fit_transform(word_vectors)

    # plt.cla()
    fig = plt.figure()
    top_word_idx = vocab.text2idx(" ".join(vocab.most_common(1000)))
    plt.plot(coords[top_word_idx, 0], coords[top_word_idx, 1], 'o', markerfacecolor='none', markeredgecolor='k',
             alpha=0.5, markersize=3)

    for i in tqdm(top_word_idx):
        plt.annotate(vocab.idx2text([i])[0],
                     xy=(coords[i, 0], coords[i, 1]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom',
                     fontsize=5)
    #plt.show()
    plt.savefig('tsne.pdf')


if __name__ == "__main__":
    main_freq()
