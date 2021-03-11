import logging
import random

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

from Vocabulary import Vocabulary
from build_freq_vectors import compute_cooccurrence_matrix, plot_word_vectors_tsne

random.seed(42)
np.random.seed(42)

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


class UnimplementedFunctionError(Exception):
    pass


################################################################################################
# Main Skeleton Code Driver
################################################################################################
def main_glove():
    logging.info("Loading dataset")
    dataset = load_dataset("ag_news")
    dataset_text = [r['text'] for r in dataset['train']]
    dataset_labels = [r['label'] for r in dataset['train']]

    logging.info("Building vocabulary")
    vocab = Vocabulary(dataset_text)
    plt.close()
    plt.pause(0.01)

    logging.info("Computing count matrix")
    C, N = compute_cooccurrence_matrix([doc['text'] for doc in dataset['train']], vocab)

    ########################################################################
    # Hyperparmeters and Optimization Variables
    ########################################################################

    d = 32  # dimensionality of the vectors
    B = 1024  # batch size (in number of word pairs)
    maxEpoch = 30  # maximum number of epochs
    learningRate = 0.1  # learning rate / step size for SGD
    clip = 50  # gradient clip value
    m = 0.9  # moment parameter

    # Initialize word vectors randomly
    wordvecs = np.random.rand(vocab.size, d)
    contextvecs = np.random.rand(vocab.size, d)

    # Initalize biases to approximate occurence frequency
    wordbiases = np.log(np.mean(C, axis=1))[:, np.newaxis]
    contextbiases = np.log(np.mean(C, axis=0))[:, np.newaxis]

    # Create empty variables to store gradients for momentum
    wordvecs_momentum = np.zeros((vocab.size, d))
    contextvecs_momentum = np.zeros((vocab.size, d))
    wordbiases_momentum = np.zeros((vocab.size, 1))
    contextbiases_momentum = np.zeros((vocab.size, 1))

    # Get all non-zero word-pair co-occurences
    idx = np.vstack(np.nonzero(C)).T
    shuf = list(range(idx.shape[0]))
    logging.info("{} non-zero entries in the count matrix".format(idx.shape[0]))

    logging.info("Starting GloVe optimization")
    for epoch in range(maxEpoch):

        # Shuffle order that word-pair will be seen
        np.random.shuffle(shuf)

        # Start an epoch
        logging.info("Epoch {} / {}: learning rate = {}".format(epoch + 1, maxEpoch, learningRate))
        loss = 0

        # for each batch
        for b in range(idx.shape[0] // B + 1):

            # Get indexes for word-pairs in the batch
            shuf_idx = shuf[b * B:min((b + 1) * B, idx.shape[0])]
            bSize = min((b + 1) * B, idx.shape[0]) - b * B
            i = idx[shuf_idx, 0]
            j = idx[shuf_idx, 1]

            # Get word and context vectors for pairs in the batch
            w_batch = wordvecs[i, :]
            c_batch = contextvecs[j, :]

            # Compute f(C_i,j) for i,j pairs in batch (Bx1 dimensional vector)
            fval = (np.minimum(1, C[i, j] / 100) ** 0.75)[:, np.newaxis]
            assert fval.shape == (bSize, 1), "Incorrect dimension for the vector of weights, should be (batch_size x 1)"

            # Compute error term as (w_i^T c_j + b_i + b_i - log(C_ij)) for each i,j pair in the batch. (Bx1 dimensional vector)
            error = (np.sum(np.multiply(w_batch, c_batch), axis=1)[:, np.newaxis] + wordbiases[i] + contextbiases[
                j] - np.log(C[i, j])[:, np.newaxis])
            assert error.shape == (bSize, 1), "Incorrect dimension for the vector of errors, should be (batch_size x 1)"

            # Combine the overall objective loss
            loss += np.sum(fval * np.square(error))

            ########################################################################
            # Task 3.2
            ########################################################################

            # write expressions using numpy to implement the gradients you derive in 3.1.
            wordvecs_grad = np.zeros((bSize, d))
            wordbiases_grad = np.zeros((bSize, 1))
            contextvecs_grad = np.zeros((bSize, d))
            contextbiases_grad = np.zeros((bSize, 1))

            wordvecs_grad = np.multiply(np.multiply(fval, c_batch), error)
            wordbiases_grad = np.multiply(fval, error)
            contextvecs_grad = np.multiply(np.multiply(fval, w_batch), error)
            contextbiases_grad = np.multiply(fval, error)

            ########################################################################

            assert wordvecs_grad.shape == (bSize, d), "Incorrect dimension, should be (batch_size x d)"
            assert contextvecs_grad.shape == (bSize, d), "Incorrect dimension, should be (batch_size x d)"
            assert wordbiases_grad.shape == (bSize, 1), "Incorrect dimension, should be (batch_size x 1)"
            assert contextbiases_grad.shape == (bSize, 1), "Incorrect dimension, should be (batch_size x 1)"

            # Update momentum
            wordvecs_momentum *= m
            wordbiases_momentum *= m
            contextvecs_momentum *= m
            contextbiases_momentum *= m

            wordvecs_momentum[i, :] += (1 - m) * wordvecs_grad
            wordbiases_momentum[i] += (1 - m) * wordbiases_grad

            contextvecs_momentum[j, :] += (1 - m) * contextvecs_grad
            contextbiases_momentum[j] += (1 - m) * contextbiases_grad

            # Apply the gradient step (with momentum and gradient clipping)
            wordvecs[i, :] -= np.maximum(np.minimum(learningRate * wordvecs_momentum[i, :], clip), -clip)
            wordbiases[i] -= np.maximum(np.minimum(learningRate * wordbiases_momentum[i], clip), -clip)

            contextvecs[j, :] -= np.maximum(np.minimum(learningRate * contextvecs_momentum[j, :], clip), -clip)
            contextbiases[j] -= np.maximum(np.minimum(learningRate * contextbiases_momentum[j], clip), -clip)

            if b % 100 == 0 and b > 0:
                logging.info(
                    "Iter {} / {}: avg. loss over last 100 batches = {}".format(b, idx.shape[0] // B, loss / (B * 100)))
                loss = 0

    logging.info("Building TSNE plot")
    plot_word_vectors_tsne(wordvecs, vocab)


if __name__ == "__main__":
    main_glove()
