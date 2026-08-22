import math
import torch
import torch.nn as nn


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        # Skalowanie o sqrt(d_model) zapobiega zdominowaniu słów przez pozycje:
        return (self.token_embedding(x) * math.sqrt(self.embedding_dim)) + self.position_embedding(positions)