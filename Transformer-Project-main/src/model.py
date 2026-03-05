from utilities import PositionalEncoding, PositionwiseFeedForward, Embeddings
import torch.nn as nn
import copy
from attention import MultiHeadedAttention
from encoder import Encoder, EncoderLayer


def make_model(
    src_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, seq_length=128
):
    """Helper: Construct a model from hyperparameters."""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = AttentionClassifier(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        ClassificationHead(d_model, seq_length),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class AttentionClassifier(nn.Module):
    """
    Encoder-Classifier architecture. This model is a transformer-attention
    model adapted to a binary classification task.
    """

    def __init__(self, encoder, head, src_embed):
        super(AttentionClassifier, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.head = head

    def forward(self, src):
        "Take in and process masked src and target sequences."
        x = self.encoder(self.src_embed(src))
        x = self.head(x)
        return x


class ClassificationHead(nn.Module):
    """
    Classification head. This module takes the output of the encoder and
    performs a classification task.
    """

    def __init__(self, d_model, seq_length):
        super(ClassificationHead, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(d_model * seq_length, 128)
        self.linear2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x).relu()
        x = self.linear2(x)
        return x
