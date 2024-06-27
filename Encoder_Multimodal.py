import math
import os
from typing import Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from Encoder_seq import EmbeddingSeq, PositionalEncoding
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FushionFC(nn.Module):
    def __init__(self, graph_enc, seq_enc):
        super().__init__()
        self.graph_enc = graph_enc
        self.seq_enc = seq_enc
        self.pool = global_mean_pool
        self.projection_head = nn.Sequential(nn.Linear(512, 512),
                                             nn.ReLU(inplace=True),
                                             nn.Linear(512, 512))
        self.FC_layer = nn.Sequential(nn.Linear(512, 256), nn.ReLU(),
                                      nn.Linear(256, 24), nn.ReLU(),
                                      nn.Linear(24, 1))

        for param in self.graph_enc.parameters():
            param.requires_grad = False

        for param in self.seq_enc.parameters():
            param.requires_grad = False

    def forward(self, graph_data, seq_data):
        output_graph = self.graph_enc(graph_data.x,
                                      graph_data.edge_index,
                                      graph_data.edge_attr,
                                      graph_data.batch)
        output_graph = self.pool(output_graph, graph_data.batch)
        output_graph = self.projection_head(output_graph)

        amino_acid = seq_data['amino_acid_indices'].clone().detach().to(DEVICE)
        binding = seq_data['binding_indices'].clone().detach().to(DEVICE)
        mask = seq_data['attention_mask'].clone().detach().to(DEVICE)
        output_seq = self.seq_enc(amino_acid, binding, mask)
        output = self.FC_layer(output_seq + output_graph)

        return output, output_seq

    def from_pretrained(self, model_file_seq, model_file_graph):
        self.graph_enc.gnn.load_state_dict(torch.load(model_file_graph))
        self.seq_enc.load_state_dict(torch.load(model_file_seq))


class AddNorm(nn.Module):
    def __init__(self, normalize_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.normalize = nn.LayerNorm(normalize_shape)

    def forward(self, X, Y):
        Y = self.dropout(Y) + X
        return self.normalize(Y)


class FFN(nn.Module):
    def __init__(self, ffn_inputs, ffn_hiddens=1024, ffn_outputs=512):
        super().__init__()
        self.dense1 = nn.Linear(ffn_inputs, ffn_hiddens)
        self.dense2 = nn.Linear(ffn_hiddens, ffn_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        X = self.dense2(self.relu(self.dense1(X)))
        return X.to(DEVICE)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_hiddens, num_heads, dropout=0.0, desc='enc'):
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

    def forward(self, queries, keys, values):
        # get matrices of q, k, v
        q, k, v = self.Q(queries), self.K(keys), self.V(values)
        q_split = q.unsqueeze(1).chunk(self.num_heads, dim=-1)
        k_split = k.unsqueeze(1).chunk(self.num_heads, dim=-1)
        v_split = v.unsqueeze(1).chunk(self.num_heads, dim=-1)

        q_stack = torch.stack(q_split, dim=1)
        k_stack = torch.stack(k_split, dim=1)
        v_stack = torch.stack(v_split, dim=1)

        score = torch.matmul(q_stack, k_stack.permute(0, 1, 3, 2))
        score = score / (k_stack.size()[-1] ** 0.5)

        score = F.softmax(score, dim=-1)
        a = torch.matmul(score, v_stack)
        a = torch.reshape(a.permute(0, 1, 3, 2), shape=(q.size(0), q.size(1)))
        a += queries

        return a


class MultimodalEncoderBlock(nn.Module):
    def __init__(self, num_hiddens, num_heads, dropout_rate):
        super().__init__()
        self.self_attention = MultiHeadAttention(num_hiddens, num_heads)
        self.cross_attention = MultiHeadAttention(num_hiddens, num_heads)

        self.ffn = FFN(num_hiddens)

        self.layer_norm1 = AddNorm(num_hiddens, dropout=dropout_rate)
        self.layer_norm2 = AddNorm(num_hiddens, dropout=dropout_rate)
        self.layer_norm3 = AddNorm(num_hiddens, dropout=dropout_rate)

    def forward(self, modality_1, modality_2):
        attn_output1 = self.self_attention(modality_1, modality_1, modality_1)
        modality_1 = self.layer_norm1(modality_1, attn_output1)

        attn_output2 = self.cross_attention(modality_1, modality_2, modality_2)
        modality_2 = self.layer_norm2(modality_2, attn_output2)

        output = self.ffn(modality_2)
        output = self.layer_norm3(modality_2, output)

        return output


class MultimodalEncoderMainSeq(nn.Module):
    def __init__(self, graph_enc, seq_enc, num_blocks):
        super().__init__()
        self.graph_enc = graph_enc
        self.seq_enc = seq_enc
        self.pool = global_mean_pool
        self.projection_head = nn.Sequential(nn.Linear(512, 512),
                                             nn.ReLU(inplace=True),
                                             nn.Linear(512, 512))
        self.attention_layer = nn.ModuleList(MultimodalEncoderBlock(num_hiddens=512,
                                                                    num_heads=4,
                                                                    dropout_rate=0.1)
                                             for _ in range(num_blocks))
        self.FC_layer = nn.Sequential(nn.Linear(512, 256), nn.ReLU(),
                                      nn.Linear(256, 24), nn.ReLU(),
                                      nn.Linear(24, 1))

        for param in self.graph_enc.parameters():
            param.requires_grad = False

        for param in self.seq_enc.parameters():
            param.requires_grad = False

    def forward(self, graph_data, seq_data):
        output_graph = self.graph_enc(graph_data.x,
                                      graph_data.edge_index,
                                      graph_data.edge_attr,
                                      graph_data.batch)
        output_graph = self.pool(output_graph, graph_data.batch)
        output_graph = self.projection_head(output_graph)

        amino_acid = seq_data['amino_acid_indices'].clone().detach().to(DEVICE)
        binding = seq_data['binding_indices'].clone().detach().to(DEVICE)
        mask = seq_data['attention_mask'].clone().detach().to(DEVICE)
        output_seq = self.seq_enc(amino_acid, binding, mask)
        for layer in self.attention_layer:
            output_multimodal = layer(output_graph, output_seq)

        output = self.FC_layer(output_multimodal)
        return output, output_seq

    def from_pretrained(self, model_file_seq, model_file_graph):
        self.graph_enc.gnn.load_state_dict(torch.load(model_file_graph))
        self.seq_enc.load_state_dict(torch.load(model_file_seq))


class BridgeTowerBlock(nn.Module):
    def __init__(self, num_hiddens, num_heads, dropout_rate):
        super().__init__()
        self.self_attetion_1 = MultiHeadAttention(num_hiddens=num_hiddens, num_heads=num_heads)
        self.self_attetion_2 = MultiHeadAttention(num_hiddens=num_hiddens, num_heads=num_heads)

        self.cross_attetion_1 = MultiHeadAttention(num_hiddens=num_hiddens, num_heads=num_heads)
        self.cross_attetion_2 = MultiHeadAttention(num_hiddens=num_hiddens, num_heads=num_heads)

        self.bridge_layer_1 = nn.Linear(num_hiddens, num_hiddens)
        self.bridge_layer_2 = nn.Linear(num_hiddens, num_hiddens)

        self.ffn_1 = FFN(num_hiddens)
        self.ffn_2 = FFN(num_hiddens)

        self.AddNorm1 = AddNorm(num_hiddens, dropout=dropout_rate)
        self.AddNorm2 = AddNorm(num_hiddens, dropout=dropout_rate)
        self.AddNorm3 = AddNorm(num_hiddens, dropout=dropout_rate)
        self.AddNorm4 = AddNorm(num_hiddens, dropout=dropout_rate)
        self.AddNorm5 = AddNorm(num_hiddens, dropout=dropout_rate)
        self.AddNorm6 = AddNorm(num_hiddens, dropout=dropout_rate)

    def forward(self, modality_1, modality_2):
        output_1 = self.bridge_layer_1(modality_1)
        output_2 = self.bridge_layer_2(modality_2)

        output_attn_1 = self.self_attetion_1(output_1, output_1, output_1)
        output_attn_2 = self.self_attetion_2(output_2, output_2, output_2)

        modality_1 = self.AddNorm1(modality_1, output_attn_1)
        modality_2 = self.AddNorm2(modality_2, output_attn_2)

        output_attn_1 = self.cross_attetion_1(modality_1, modality_2, modality_2)
        output_attn_2 = self.cross_attetion_2(modality_2, modality_1, modality_1)

        modality_1 = self.AddNorm3(modality_1, output_attn_1)
        modality_2 = self.AddNorm4(modality_2, output_attn_2)

        output1 = self.ffn_1(modality_1)
        output2 = self.ffn_2(modality_2)

        output1 = self.AddNorm5(modality_1, output1)
        output2 = self.AddNorm6(modality_2, output2)

        return output1, output2


class BridgeTowerEncoder(nn.Module):
    def __init__(self, graph_enc, seq_enc, num_blocks):
        super().__init__()
        self.graph_enc = graph_enc
        self.pool = global_mean_pool
        self.projection_head = torch.nn.Sequential(torch.nn.Linear(512, 512),
                                                   torch.nn.ReLU(inplace=True),
                                                   torch.nn.Linear(512, 512))
        self.seq_enc = seq_enc

        self.bridge_tower = nn.ModuleList(BridgeTowerBlock(num_hiddens=512,
                                                           num_heads=4,
                                                           dropout_rate=0.1)
                                          for _ in range(num_blocks))

        self.FC_layer = nn.Sequential(nn.Linear(512, 256), nn.ReLU(),
                                      nn.Linear(256, 24), nn.ReLU(),
                                      nn.Linear(24, 1))

        for param in self.graph_enc.parameters():
            param.requires_grad = False

        for param in self.seq_enc.parameters():
            param.requires_grad = False

    def forward(self, graph_data, seq_data):
        output_graph = self.graph_enc(graph_data.x,
                                      graph_data.edge_index,
                                      graph_data.edge_attr,
                                      graph_data.batch)
        output_graph = self.pool(output_graph, graph_data.batch)
        output_graph = self.projection_head(output_graph)

        amino_acid = seq_data['amino_acid_indices'].clone().detach().to(DEVICE)
        binding = seq_data['binding_indices'].clone().detach().to(DEVICE)
        mask = seq_data['attention_mask'].clone().detach().to(DEVICE)
        output_seq = self.seq_enc(amino_acid, binding, mask)

        for layer in self.bridge_tower:
            output_graph, output_seq = layer(output_graph, output_seq)
        last_hidden_state = output_graph + output_seq
        output = self.FC_layer(last_hidden_state)
        return output, last_hidden_state

    def from_pretrained(self, model_file_seq, model_file_graph):
        self.graph_enc.gnn.load_state_dict(torch.load(model_file_graph))
        self.seq_enc.load_state_dict(torch.load(model_file_seq))

























