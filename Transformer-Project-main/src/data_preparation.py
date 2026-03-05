import numpy as np
import nltk
from nltk.corpus import stopwords
import torch
from torch.utils.data import TensorDataset, DataLoader
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


def data_processing(data, batch_size=128, seq_length=256, test_size=0.2):
    """
    Function that processes the data and returns the train and test dataloaders, as well as the vocabulary size.
    """
    # get all processed reviews
    reviews = data.processed.values
    # merge into single variable, separated by whitespaces
    words = " ".join(reviews)
    # obtain list of words
    words = words.split()

    # build vocabulary
    counter = Counter(words)
    vocab = sorted(counter, key=counter.get, reverse=True)
    int2word = dict(enumerate(vocab, 1))
    int2word[0] = "<PAD>"
    word2int = {word: id for id, word in int2word.items()}

    # encode words
    reviews_enc = [
        [word2int[word] for word in review.split()] for review in tqdm(reviews)
    ]

    features = pad_features(
        reviews_enc, pad_id=word2int["<PAD>"], seq_length=seq_length
    )

    assert len(features) == len(reviews_enc)
    assert len(features[0]) == seq_length

    # get labels as numpy
    labels = data.label.to_numpy()

    # train test split
    assert 0 < test_size < 1
    train_size = 1 - test_size

    # make train set
    split_id = int(len(features) * train_size)
    train_x, test_x = features[:split_id], features[split_id:]
    train_y, test_y = labels[:split_id], labels[split_id:]

    # print out the shape
    print("Feature Shapes:")
    print("===============")
    print("Train set: {}".format(train_x.shape))
    print("Test set: {}".format(test_x.shape))

    # create tensor datasets
    train_set = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    test_set = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    # create dataloaders
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=batch_size)

    return train_loader, test_loader, len(word2int)
