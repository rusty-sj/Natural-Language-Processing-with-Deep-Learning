# Basic python imports for logging and sequence generation
import itertools
import logging
import random

from torch.nn import LSTM, Linear

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# Imports for Pytorch for the things we need
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

# Imports for plotting our result curves
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

# Set random seed for python and torch to enable reproducibility (at least on the same hardware)
random.seed(42)
torch.manual_seed(42)

# Determine if a GPU is available for use, define as global variable
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Main Driver Loop
def main():
    # Build the model and put it on the GPU
    logging.info("Building model")
    model = ParityLSTM()
    model.to(dev)  # move to GPU if cuda is enabled

    logging.info("Training model")
    maximum_training_sequence_length = 5
    train = Parity(split='train', max_length=maximum_training_sequence_length)
    train_loader = DataLoader(train, batch_size=100, shuffle=True, collate_fn=pad_collate)
    train_model(model, train_loader)

    logging.info("Running generalization experiment")
    runParityExperiment(model, maximum_training_sequence_length)


######################################################################
# Task 2.2
######################################################################

# Implement a LSTM model for the parity task.

class ParityLSTM(torch.nn.Module):

    # __init__ builds the internal components of the model (presumably an LSTM and linear layer for classification)
    # The LSTM should have hidden dimension equal to hidden_dim

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = LSTM(input_size=1, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.linear = Linear(in_features=hidden_dim, out_features=2)

    # forward runs the model on an B x max_length x 1 tensor and outputs a B x 2 tensor representing a score for
    # even/odd parity for each element of the batch
    #
    # Inputs:
    #   x -- a batch_size x max_length x 1 binary tensor. This has been padded with zeros to the max length of
    #        any sequence in the batch.
    #   s -- a batch_size x 1 list of sequence lengths. This is useful for ensuring you get the hidden state at
    #        the end of a sequence, not at the end of the padding
    #
    # Output:
    #   out -- a batch_size x 2 tensor of scores for even/odd parity

    def forward(self, x, s):
        x_unsqueezed = torch.unsqueeze(x, 2)
        x_pack = pack_padded_sequence(input=x_unsqueezed, lengths=s, batch_first=True, enforce_sorted=False).to(dev)
        out_pack, ht = self.lstm(x_pack)
        return self.linear(ht[0][-1])

    def __str__(self):
        return "LSTM-" + str(self.hidden_dim)


######################################################################


# This function evaluate a model on binary strings ranging from length 1 to 20.
# A plot is saved in the local directory showing accuracy as a function of this length
def runParityExperiment(model, max_train_length):
    logging.info("Starting parity experiment with model: " + str(model))
    lengths = []
    accuracy = []

    logging.info("Evaluating over strings of length 1-20.")
    k = 1
    val_acc = 1
    while k <= 20:
        val = Parity(split='val', max_length=k)
        val_loader = DataLoader(val, batch_size=1000, shuffle=False, collate_fn=pad_collate)
        val_loss, val_acc = validation_metrics(model, val_loader)
        lengths.append(k)
        accuracy.append(val_acc)

        logging.info("length=%d val accuracy %.3f" % (k, val_acc))
        k += 1

    plt.plot(lengths, accuracy)
    plt.axvline(x=max_train_length, c="k", linestyle="dashed")
    plt.xlabel("Binary String Length")
    plt.ylabel("Accuracy")
    plt.title('LSTM-64 Parity Generalization')
    plt.savefig(str(model) + '_parity_generalization.png')


# Dataset of binary strings, during training generates up to length max_length
# during validation, just create sequences of max_length
class Parity(Dataset):

    def __init__(self, split="train", max_length=4):
        if split == "train":
            self.data = []
            for i in range(1, max_length + 1):
                self.data += [torch.FloatTensor(seq) for seq in itertools.product([0, 1], repeat=i)]
        else:
            self.data = [torch.FloatTensor(seq) for seq in itertools.product([0, 1], repeat=max_length)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        x = self.data[idx]
        y = x.sum() % 2
        return x, y

    # Function to enable batch loader to concatenate binary strings of different lengths and pad them


def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy = torch.LongTensor(yy)

    return xx_pad, yy, x_lens


# Basic training loop for cross entropy loss
def train_model(model, train_loader, epochs=2000, lr=0.003):
    # Define a cross entropy loss function
    crit = torch.nn.CrossEntropyLoss()

    # Collect all the learnable parameters in our model and pass them to an optimizer
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    # Adam is a version of SGD with dynamic learning rates
    # (tends to speed convergence but often worse than a well tuned SGD schedule)
    optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=0.00001)

    # Main training loop over the number of epochs
    for i in range(epochs):

        # Set model to train mode so things like dropout behave correctly
        model.train()
        sum_loss = 0.0
        total = 0
        correct = 0

        # for each batch in the dataset
        for j, (x, y, l) in enumerate(train_loader):
            # push them to the GPU if we are using one
            x = x.to(dev)
            y = y.to(dev)

            # predict the parity from our model
            y_pred = model(x, l)

            # compute the loss with respect to the true labels
            loss = crit(y_pred, y)

            # zero out the gradients, perform the backward pass, and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute loss and accuracy to report epoch level statitics
            pred = torch.max(y_pred, 1)[1]
            correct += (pred == y).float().sum()
            sum_loss += loss.item() * y.shape[0]
            total += y.shape[0]
        if i % 10 == 0:
            logging.info("epoch %d train loss %.3f, train acc %.3f" % (
                i, sum_loss / total, correct / total))  # , val_loss, val_acc))


def validation_metrics(model, loader):
    # set the model to evaluation mode to turn off things like dropout
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    crit = torch.nn.CrossEntropyLoss()
    for i, (x, y, l) in enumerate(loader):
        x = x.to(dev)
        y = y.to(dev)
        y_hat = model(x, l)

        loss = crit(y_hat, y)
        pred = torch.max(y_hat, 1)[1]
        correct += (pred == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item() * y.shape[0]

    return sum_loss / total, correct / total


if __name__ == "__main__":
    main()
