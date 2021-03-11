import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from spacy.lang.en import English
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

BATCH_SIZE = 64
EMBEDDING_DIM = 100
HIDDEN_DIM = 32
LR_RATE = 0.001
WT_DECAY = 0.00001
N_LAYERS = 1
N_EPOCHS = 15

TEXT = data.Field(
    lower=True,
    fix_length=EMBEDDING_DIM,
    batch_first=True
)
UD_TAGS = data.Field(
    sequential=True,
    batch_first=True,
    fix_length=EMBEDDING_DIM,
    is_target=True,
)

fields = (("text", TEXT), ("udtags", UD_TAGS))
d_train, d_val, d_test = datasets.UDPOS.splits(fields)

TEXT.build_vocab(d_train, vectors=[GloVe(name="6B", dim="100")], min_freq=2, unk_init=torch.Tensor.normal_)
UD_TAGS.build_vocab(d_train)

train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (d_train, d_val, d_test),
    batch_size=BATCH_SIZE,
    device=dev)
INPUT_DIM = len(TEXT.vocab)
OUTPUT_DIM = len(UD_TAGS.vocab)
PAD_TOKEN_IDX = TEXT.vocab.stoi[TEXT.pad_token]


class BiLSTM(torch.nn.Module):
    def __init__(self, input_dim, embedding_dim, output_dim, pad_token_idx, hidden_dim=64, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_token_idx)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=n_layers,
                            batch_first=True,
                            bidirectional=True)
        self.linear = nn.Linear(in_features=2 * hidden_dim, out_features=output_dim)
        self.loss_criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_IDX).to(dev)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LR_RATE, weight_decay=WT_DECAY)
        self.pad_token_idx = pad_token_idx

    def forward(self, text):
        # text [max_length x B]
        embedded = self.embedding(text)  # [max_length x B x embedding_dim]
        outs, _ = self.lstm(embedded)
        return self.linear(outs)  # [max_length x B x output_dim]

    def compute_accuracy(self, y_pred, y):
        max_preds = y_pred.argmax(dim=1, keepdim=True)
        non_padded = torch.nonzero(y != self.pad_token_idx)
        hits = max_preds[non_padded].squeeze(1).eq(y[non_padded])
        return hits.sum().to(dev) / torch.FloatTensor([y[non_padded].shape[0]]).to(dev)

    def compute_metrics(self, y_pred, y):
        loss = self.loss_criterion(y_pred, y)
        accu = self.compute_accuracy(y_pred=y_pred, y=y)
        return loss, accu

    def step(self, batch):
        # text [max_len x B]
        text, tags = batch.text, batch.udtags
        # print(text, tags)
        self.optimizer.zero_grad()

        y_pred = self.forward(text)  # [max_len x B x output_dim]
        y_pred = y_pred.view(-1, y_pred.shape[-1])
        tags = tags.view(-1)

        loss, accu = self.compute_metrics(y_pred=y_pred, y=tags)
        loss.backward()

        self.optimizer.step()
        return loss.item(), accu.item()


def train(model, iterator):
    epoch_loss, epoch_accu = 0.0, 0.0
    model.train()

    for batch in iterator:
        loss, accu = model.step(batch=batch)
        epoch_loss += loss
        epoch_accu += accu

    epoch_loss /= len(iterator)
    epoch_accu /= len(iterator)
    return epoch_loss, epoch_accu


def evaluate(model, iterator):
    epoch_loss, epoch_accu = 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text, tags = batch.text, batch.udtags
            y_pred = model(text)
            y_pred = y_pred.view(-1, y_pred.shape[-1])
            tags = tags.view(-1)
            loss, accu = model.compute_metrics(y_pred=y_pred, y=tags)
            epoch_loss += loss.item()
            epoch_accu += accu.item()
        epoch_loss /= len(iterator)
        epoch_accu /= len(iterator)
    return epoch_loss, epoch_accu


def tag_sentence(model, sentence):
    nlp = English()
    # Create a Tokenizer with the default settings for English
    # including punctuation rules and exceptions
    tokens = [token.text for token in nlp.tokenizer(sentence)]
    tokens = [token.lower() for token in tokens]
    numeralized_tokens = [TEXT.vocab.stoi[token] for token in tokens]
    model_input = torch.LongTensor(numeralized_tokens).unsqueeze(-1).to(dev)
    preds = model(model_input).argmax(-1)
    pred_tags = [UD_TAGS.vocab.itos[pred.item()] for pred in preds]
    for index, token in enumerate(tokens):
        print(f"{token}\t=>\t{pred_tags[index]}")
    print("---------------------------")


def visualizeSentenceWithTags(example):
    print("Token" + "".join([" "] * (15)) + "POS Tag")
    print("---------------------------------")
    for w, t in zip(example["text"], example["tags"]):
        print(w + "".join([" "] * (20 - len(w))) + t)


def datasets_lookup():
    visualizeSentenceWithTags(vars(d_train.examples[997]))
    all_words = []
    all_tags = []

    for example in d_train.examples:
        print(example.text)
        # print(example.tags)
        # visualizeSentenceWithTags(vars(example))
        all_words += example.text
        all_tags += example.tags

    # Plot POS tags distribution
    labels, values = zip(*Counter(all_tags).most_common())
    print(labels, values)
    print(len(all_words))
    indexes = np.arange(len(labels))
    width = 1
    fig = plt.figure()
    plt.bar(indexes, values, width)
    plt.xticks(indexes + width * 0.5, labels, rotation=45)
    plt.title('POS tags frequencies')
    # plt.show()
    # plt.savefig('udpos.png')


if __name__ == '__main__':
    # datasets_lookup()
    model = BiLSTM(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_TOKEN_IDX, HIDDEN_DIM, N_LAYERS)
    model.embedding.weight.data[PAD_TOKEN_IDX] = torch.zeros(EMBEDDING_DIM)
    model = model.to(dev)
    tr_losses, tr_accuracies = [], []
    val_losses, val_accuracies = [], []
    least_val_loss = float('inf')

    for epoch in range(N_EPOCHS):
        tr_loss, tr_acc = train(model, train_iter)
        tr_losses.append(tr_loss)
        tr_accuracies.append(tr_acc)
        print(f"epoch={epoch + 1}\t tr_loss={tr_loss:.2f}\t tr_accu={tr_acc:.2f}")
        val_loss, val_acc = evaluate(model, val_iter)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        if val_loss < least_val_loss:
            least_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
        print(f"epoch={epoch + 1}\t val_loss={val_loss:.2f}\t val_accu={val_acc:.2f}")

    plt.figure()
    plt.plot(np.arange(start=1, stop=N_EPOCHS + 1), tr_accuracies, "-b", label="Train Accuracy")
    plt.plot(np.arange(start=1, stop=N_EPOCHS + 1), val_accuracies, "-r", label="Val Accuracy")
    plt.legend()
    plt.ylim(0.4, 1)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    # plt.savefig('31.png')
    # files.download("31.png")

    plt.figure()
    plt.plot(np.arange(start=1, stop=N_EPOCHS + 1), tr_losses, "-b", label="Train Loss")
    plt.plot(np.arange(start=1, stop=N_EPOCHS + 1), val_losses, "-r", label="Val Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.savefig('32.png')
    # files.download("32.png")

    model.load_state_dict(torch.load('best_model.pt'))
    test_loss, test_acc = evaluate(model, test_iter)
    print(f"test_loss={test_loss:.2f}\t test_accu={test_acc:.2f}")

    print("---------------------------")
    tag_sentence(model, "The old man the boat.")
    tag_sentence(model, "The complex houses married and single soldiers and their families.")
    tag_sentence(model, "The man who hunts ducks out on weekends.")
