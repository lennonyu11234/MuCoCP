import torch
from torch import nn, optim
from torch.nn import functional as F
from torchinfo import summary
import math
from torch.utils.data import Dataset, DataLoader, random_split
import tqdm
import matplotlib.pyplot as plt
import csv

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EmbeddingSeq(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)

    def forward(self, X):
        X = self.embed(X).to(DEVICE)
        return X


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, dropout=0.1, max_len=50):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        PE = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        PE[:, 0::2] = torch.sin(position * div_term)
        PE[:, 1::2] = torch.cos(position * div_term)
        PE = PE.unsqueeze(0).transpose(0, 1)

        self.register_buffer('PE', PE)

    def forward(self, X):
        X = X + self.PE[:X.size(0), :]
        return X


class FFN(nn.Module):
    def __init__(self, ffn_inputs, ffn_hiddens=1024, ffn_outputs=512):
        super().__init__()
        self.dense1 = nn.Linear(ffn_inputs, ffn_hiddens)
        self.dense2 = nn.Linear(ffn_hiddens, ffn_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        X = self.dense2(self.relu(self.dense1(X)))
        return X.to(DEVICE)


class AddNorm(nn.Module):
    def __init__(self, normalize_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.normalize = nn.LayerNorm(normalize_shape)

    def forward(self, X, Y):
        Y = self.dropout(Y) + X
        return self.normalize(Y)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_hiddens=512, num_heads=4, dropout=0.01, desc='enc'):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.normalize = AddNorm(num_hiddens, dropout)
        self.desc = desc

        # define q,k,v linear layer
        self.Wq = nn.Linear(self.num_hiddens, self.num_hiddens)
        self.Wk = nn.Linear(self.num_hiddens, self.num_hiddens)
        self.Wv = nn.Linear(self.num_hiddens, self.num_hiddens)

        self.relu = nn.ReLU()
        self.Q = nn.Sequential(self.Wq, self.relu)
        self.K = nn.Sequential(self.Wk, self.relu)
        self.V = nn.Sequential(self.Wv, self.relu)

    def forward(self, queries, keys, values, attention_mask):
        # get matrices of q, k, v
        q, k, v = self.Q(queries), self.K(keys), self.V(values)
        # 得到经多头切分后的矩阵 shape:(batch_size, len_seq, d_model/num_heads)[在最后一维切分]
        q_split = torch.chunk(q, self.num_heads, dim=-1)
        k_split = torch.chunk(k, self.num_heads, dim=-1)
        v_split = torch.chunk(v, self.num_heads, dim=-1)
        # 将他们在第二维（new）堆叠  shape:(batch_size, num_heads, len_seq, d_model/num_heads)
        q_stack = torch.stack(q_split, dim=1)
        k_stack = torch.stack(k_split, dim=1)
        v_stack = torch.stack(v_split, dim=1)
        # get attention score
        score = torch.matmul(q_stack, k_stack.permute(0, 1, 3, 2))
        score = score / (k_stack.size()[-1] ** 0.5)

        if self.desc == 'enc':
            enc_attention_mask = (attention_mask.unsqueeze(1)).unsqueeze(3).repeat(1, self.num_heads, 1, 1)
            score.masked_fill_(enc_attention_mask, -1e9)

        score = F.softmax(score, dim=-1)
        a = torch.matmul(score, v_stack)
        a = torch.reshape(a.permute(0, 1, 3, 2), shape=(q.size(0), q.size(1), q.size(2)))
        a += queries

        return a


class EncoderBlock(nn.Module):
    def __init__(self, num_hiddens=512, num_heads=4, dropout=0.01):
        super().__init__()
        self.attention = MultiHeadAttention(num_hiddens=num_hiddens,
                                            num_heads=num_heads,
                                            dropout=dropout,
                                            desc='enc')
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.addnorm2 = AddNorm(num_hiddens, dropout)
        self.FFN = FFN(ffn_inputs=num_hiddens)

    def forward(self, X, attention_mask):
        Y = self.addnorm1(X, self.attention(X, X, X, attention_mask))
        outputs = self.addnorm2(Y, self.FFN(Y))
        return outputs.to(DEVICE)


class SeqEncoder(nn.Module):
    def __init__(self, seq_vocab, bind_vocab, num_hiddens=512):
        super().__init__()
        self.embeddingseq = EmbeddingSeq(seq_vocab, num_hiddens)
        self.embeddingbind = EmbeddingSeq(bind_vocab, num_hiddens)
        self.pe = PositionalEncoding()
        self.layers = nn.ModuleList([EncoderBlock() for _ in range(4)])

    def forward(self, seq, bind, attention_mask):
        seq_embedded = self.embeddingseq(seq)
        bind_embedded = self.embeddingbind(bind)
        bind_embedded = bind_embedded.unsqueeze(1)
        combined_embedded = seq_embedded + bind_embedded
        enc_outputs = self.pe(combined_embedded.transpose(0, 1)).transpose(0, 1)

        for layer in self.layers:
            enc_outputs = layer(enc_outputs, attention_mask)

        enc_outputs = torch.mean(enc_outputs, dim=1)
        # enc_outputs = enc_outputs.squeeze(dim=0)
        return enc_outputs.to(DEVICE)