import torch.nn as nn
import torch.nn.functional as F


class LanguageModel(nn.Module):

    def __init__(self, vocab_size, embedd_size=100, hidden_size=512, num_layers=3, embed_matrix=None):
        super(LanguageModel, self).__init__()

        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embedd_size)
        if embed_matrix is not None:
            self.embed.weight.data.copy_(embed_matrix)
        self.rnn = nn.LSTM(embedd_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, vocab_size)
        self.drop = nn.Dropout()

    def forward(self, x, h, c):
        out = self.embed(x)
        out, (h, c) = self.rnn(out, (h, c))
        out = F.relu(self.linear(self.drop(out)))
        out = self.linear2(out)
        return out, h, c
