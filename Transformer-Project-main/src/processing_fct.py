import numpy as np
import nltk
from nltk.corpus import stopwords
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
from collections import Counter

nltk.download("stopwords")
nltk.download("wordnet")
stopwords = set(stopwords.words("english"))

tqdm.pandas()


def pad_features(reviews, pad_id, seq_length=128):
    """
    Function that pads features to a given length.
    """
    # features = np.zeros((len(reviews), seq_length), dtype=int)
    features = np.full((len(reviews), seq_length), pad_id, dtype=int)

    for i, row in enumerate(reviews):
        # if seq_length < len(row) then review will be trimmed
        features[i, : len(row)] = np.array(row)[:seq_length]

    return features


def data_processing(new_input, label, seq_length=256):
    """
    Function that processes the data and returns the test dataloader, as well as the vocabulary size.
    """
    # get all processed reviews
    user_input = new_input
    # merge into single variable, separated by whitespaces
    words_dev = " ".join(user_input)
    # obtain list of words
    words_dev = words_dev.split()

    # build vocabulary
    counter_dev = Counter(words_dev)
    vocab_dev = sorted(counter_dev, key=counter_dev.get, reverse=True)
    int2word_dev = dict(enumerate(vocab_dev, 1))
    int2word_dev[0] = "<PAD>"
    word2int_dev = {word: id for id, word in int2word_dev.items()}

    # encode words
    input_enc = [
        [word2int_dev[word] for word in input.split()] for input in tqdm(new_input)
    ]

    features_dev = pad_features(
        input_enc, pad_id=word2int_dev["<PAD>"], seq_length=seq_length
    )

    assert len(features_dev) == len(input_enc)
    assert len(features_dev[0]) == seq_length

    # Test and its label
    test_x = features_dev
    # Create tendor Datasets
    test_set = TensorDataset(torch.tensor([test_x]), torch.tensor([label]))
    return test_set, len(word2int_dev)
