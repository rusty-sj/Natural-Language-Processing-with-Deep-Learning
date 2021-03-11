import os

from torch.distributions import Categorical
from torch.nn import Softmax, LogSoftmax

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'  # This is a command to reduce non-deterministic behavior in CUDA
import warnings

warnings.simplefilter("ignore", UserWarning)
import pickle
import numpy as np
import torch
import argparse
from LanguageModel import LanguageModel
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chkpt', dest='chkpt', metavar='c', default="got_language_model")
    args = parser.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Using device: {}'.format(dev))

    logging.info("Loading tokenizer and vocab from vocab.pkl")
    text_field = pickle.load(open("vocab.pkl", "rb"))
    vocab_size = len(text_field.vocab.itos)

    logging.info("Loading checkpoint {}".format(args.chkpt))
    lm = LanguageModel(vocab_size).to(dev)
    lm.load_state_dict(torch.load(args.chkpt, map_location=dev))
    lm.eval()

    p = "the night is dark and full of terrors"

    # Torch is a bit frustrating at times and some things that ought to be deterministic are not.
    # This is an attempt to resolve that, but it doesn't work 100% of the time
    torch.set_deterministic(True)
    seed = 42
    mlen = 150

    torch.manual_seed(seed)
    np.random.seed(seed)
    print("\n----------- Vanilla Sampling -----------")
    print(sample(lm, text_field, prompt=p, max_len=mlen))

    torch.manual_seed(seed)
    np.random.seed(seed)
    print("\n------- Temp-Scaled Sampling 0.0001 -------")
    print(sample(lm, text_field, prompt=p, temp=0.0001, max_len=mlen))

    torch.manual_seed(seed)
    np.random.seed(seed)
    print("\n------- Temp-Scaled Sampling 100 --------")
    print(sample(lm, text_field, prompt=p, temp=100, max_len=mlen))

    torch.manual_seed(seed)
    np.random.seed(seed)
    print("\n----------- Top-k Sampling 1 -----------")
    print(sample(lm, text_field, prompt=p, k=1, max_len=mlen))

    torch.manual_seed(seed)
    np.random.seed(seed)
    print("\n----------- Top-k Sampling 20 -----------")
    print(sample(lm, text_field, prompt=p, k=20, max_len=mlen))

    torch.manual_seed(seed)
    np.random.seed(seed)
    print("\n----------- Top-p Sampling 0.001 -----------")
    print(sample(lm, text_field, prompt=p, p=0.001, max_len=mlen))

    torch.manual_seed(seed)
    np.random.seed(seed)
    print("\n----------- Top-p Sampling 0.75 -----------")
    print(sample(lm, text_field, prompt=p, p=0.75, max_len=mlen))

    torch.manual_seed(seed)
    np.random.seed(seed)
    print("\n----------- Top-p Sampling 1 -----------")
    print(sample(lm, text_field, prompt=p, p=1, max_len=mlen))

    torch.manual_seed(seed)
    np.random.seed(seed)
    print("\n----------- Beam Search B=1 -----------")
    print(beamsearch(lm, text_field, prompt=p, beams=1, max_len=mlen))

    torch.manual_seed(seed)
    np.random.seed(seed)
    print("\n----------- Beam Search B=10 -----------")
    print(beamsearch(lm, text_field, prompt=p, beams=10, max_len=mlen))

    torch.manual_seed(seed)
    np.random.seed(seed)
    print("\n----------- Beam Search B=50 -----------")
    print(beamsearch(lm, text_field, prompt=p, beams=50, max_len=mlen))

    print()


############################################################################################
# TASK 2.1
############################################################################################

def beamsearch(model, text_field, beams=5, prompt="", max_len=50):
    hidden_size = model._modules['rnn'].hidden_size
    n_layers = model._modules['rnn'].num_layers
    h_t_prev = torch.zeros(n_layers, 1, hidden_size)
    c_t_prev = torch.zeros(n_layers, 1, hidden_size)
    w_t = text_field.process([text_field.tokenize(prompt.lower())])
    sampling_criterion = LogSoftmax(dim=1)

    s_t, h_t, c_t = model(w_t, h_t_prev, c_t_prev)
    h_t_b_prev = torch.cat([h_t] * beams, 1)
    c_t_b_prev = torch.cat([c_t] * beams, 1)

    w_t_next = sampling_criterion(s_t[-1])

    topk_vals, topk_indices = torch.topk(w_t_next, beams)
    decoded_strings = topk_indices.view(beams, 1)

    last_prob = []
    for i in range(1, max_len, 1):
        s_t, h_t_b, c_t_b = model(topk_indices, h_t_b_prev, c_t_b_prev)
        w_t_next = sampling_criterion(s_t[-1])

        cumulative_log_probs = w_t_next + topk_vals.view((beams, 1))
        topk_vals, topk_indices = torch.topk(cumulative_log_probs.view(-1), beams)
        b_indices = np.array(np.unravel_index(topk_indices.numpy(), cumulative_log_probs.shape)).T

        h_t_b_new, c_t_b_new = [], []
        for layer in range(n_layers):
            ht_layer, ct_layer = [], []
            for i, j in b_indices:
                ht_layer.append(h_t_b[layer][i])
                ct_layer.append(c_t_b[layer][i])
            h_t_b_new.append(torch.stack(ht_layer))
            c_t_b_new.append(torch.stack(ct_layer))
        h_t_b_prev = torch.stack(h_t_b_new)
        c_t_b_prev = torch.stack(c_t_b_new)

        strings = []
        for i, (r, c) in enumerate(b_indices):
            topk_indices[i] = c
            strings.append(torch.cat([decoded_strings[r], torch.tensor([c])]))

        decoded_strings = strings
        topk_indices = topk_indices.unsqueeze(0)
        last_prob = topk_vals

    decoded_strings = decoded_strings[last_prob.argmax()]
    decoded_strings = prompt + " " + reverseNumeralize(decoded_strings, text_field)
    return decoded_strings


############################################################################################
# TASK 1.1 TASK 1.2
############################################################################################

def sample(model, text_field, prompt="", max_len=50, temp=1.0, k=0, p=1):
    assert (k == 0 or p == 1), "Cannot combine top-k and top-p sampling"

    hidden_size = model._modules['rnn'].hidden_size
    n_layers = model._modules['rnn'].num_layers

    h_t_prev = torch.zeros(n_layers, 1, hidden_size)
    c_t_prev = torch.zeros(n_layers, 1, hidden_size)
    w_t = text_field.process([text_field.tokenize(prompt.lower())])

    sampling_criterion = Softmax(dim=1)
    words = []

    for i in range(max_len):
        s_t, h_t, c_t = model(w_t, h_t_prev, c_t_prev)
        w_t_next = sampling_criterion(s_t[-1] / temp)

        if k != 0:
            # Top-k Sampling
            topk_vals, topk_indices = torch.topk(w_t_next, k)
            topk_vals /= topk_vals.sum(dim=1)
            prob_dist = Categorical(topk_vals)
            w_t = topk_indices.view(-1)[prob_dist.sample()].unsqueeze(dim=0)
        elif p > 0:
            # Top-p Sampling
            sorted_wts, sorted_indices = torch.sort(w_t_next, descending=True)
            cumulative_probs = torch.cumsum(input=sorted_wts, dim=1).squeeze()
            sorted_indices_to_remove = cumulative_probs > p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove.unsqueeze(0)]
            w_t_next.squeeze()[indices_to_remove] = 0
            w_t_next /= w_t_next.sum()
            prob_dist = Categorical(w_t_next)
            w_t = prob_dist.sample().unsqueeze(0)
        else:
            # Vanilla and Temperature-Scaled Sampling
            prob_dist = Categorical(w_t_next)
            w_t = prob_dist.sample().unsqueeze(dim=0)

        words.append(w_t)
        h_t_prev = h_t
        c_t_prev = c_t

    decoded_string = prompt + " " + reverseNumeralize(torch.cat(words).squeeze(), text_field)
    return decoded_string


############################################################################################

def reverseNumeralize(numeralized_string, text_field):
    strings = [text_field.vocab.itos[i] for i in numeralized_string]
    return " ".join(strings)


if __name__ == "__main__":
    main()
