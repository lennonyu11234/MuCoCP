import argparse
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import global_mean_pool
from Encoder_graph import GNNEncoder
from Encoder_seq import SeqEncoder
from Encoder_Multimodal import BridgeTowerBlock

# =====================================[parser]==============================================
criterion = nn.BCEWithLogitsLoss()
parser = argparse.ArgumentParser(description='Train and Validation Function')
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0003)')
parser.add_argument('--dropout_ratio', type=float, default=0.1,
                    help='dropout ratio (default: 0.1)')
parser.add_argument('--random_seed', type=int, default=42,
                    help='random seed (default: 42)')
parser.add_argument('--graph_num_layers', type=int, default=5,
                    help='graph_num_layers(default: 5)')
parser.add_argument('--emb_dim', type=int, default=512,
                    help='emb_dim(default: 512)')
parser.add_argument('--num_blocks', type=int, default=4,
                    help='num_blocks(default: 4)')
parser.add_argument('--JK', type=str, default="last",
                    help='how the node features across layers are combined. last, sum, max or concat')
parser.add_argument('--gnn_type', type=str, default="gcn")
parser.add_argument('--amino_vocab_size', type=int, default=313,
                    help='amino_vocab_size(default: 313)')
parser.add_argument('--binding_vocab_size', type=int, default=63,
                    help='binding_vocab_size(default: 63)')
parser.add_argument('--mlm_probability', type=float, default=0,
                    help='mlm_probability(default: 0)')
parser.add_argument('--temp', type=float, default=0.1,
                    help='temperature(default: 0)')
parser.add_argument('--queue_size', type=int, default=1024,
                    help='queue size(default: 1024)')
parser.add_argument('--momentum', type=float, default=0.995,
                    help='momentum (default: 0.995)')


args = parser.parse_args()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PretrainMultimodal(nn.Module):
    def __init__(self, num_blocks):
        super().__init__()
        self.mlm_probability = args.mlm_probability
        self.graph_encoder = GNNEncoder(num_layer=args.graph_num_layers, emb_dim=args.emb_dim,
                                        drop_ratio=args.dropout_ratio,
                                        JK=args.JK, gnn_type=args.gnn_type).to(DEVICE)
        self.pool = global_mean_pool
        self.projection_head = torch.nn.Sequential(torch.nn.Linear(512, 512),
                                                   torch.nn.ReLU(inplace=True),
                                                   torch.nn.Linear(512, 512))
        # input_model_file = 'new_stage/Graph_Pretrain/Drop_Nodes/DropN&DropNC.pth'
        # self.graph_encoder.from_pretrained(input_model_file)
        self.seq_encoder = SeqEncoder(seq_vocab=args.amino_vocab_size,
                                      bind_vocab=args.binding_vocab_size).to(DEVICE)

        self.bridge_tower = nn.ModuleList(BridgeTowerBlock(num_hiddens=512,
                                                           num_heads=4,
                                                           dropout_rate=0.1)
                                          for _ in range(num_blocks))

        # self.graph_proj = nn.Linear(args.emb_dim, args.emb_dim)
        # self.seq_proj = nn.Linear(args.emb_dim, args.emb_dim)

        self.temp = nn.Parameter(torch.ones([]) * args.temp)
        self.queue_size = args.queue_size
        self.momentum = args.momentum
        self.itm_head = nn.Linear(args.emb_dim, 2)

        # create momentum model
        self.graph_encoder_m = GNNEncoder(num_layer=args.graph_num_layers, emb_dim=args.emb_dim,
                                          drop_ratio=args.dropout_ratio,
                                          JK=args.JK, gnn_type=args.gnn_type).to(DEVICE)
        # self.graph_proj_m = nn.Linear(args.emb_dim, args.emb_dim)

        self.seq_encoder_m = SeqEncoder(seq_vocab=args.amino_vocab_size,
                                        bind_vocab=args.binding_vocab_size).to(DEVICE)
        # self.seq_proj_m = nn.Linear(args.emb_dim, args.emb_dim)

        self.model_pairs = [[self.graph_encoder, self.graph_encoder_m],
                            [self.seq_encoder, self.seq_encoder_m]]

        self.copy_params()

        # create the queue
        self.register_buffer("graph_queue", torch.randn(args.emb_dim, self.queue_size))
        self.register_buffer("seq_queue", torch.randn(args.emb_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.graph_queue = nn.functional.normalize(self.graph_queue, dim=0)
        self.seq_queue = nn.functional.normalize(self.seq_queue, dim=0)

    def forward(self, graph_data, seq_data, alpha=0.4):
        # GSC Loss
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)
        graph_feat = self.graph_encoder(graph_data.x,
                                        graph_data.edge_index,
                                        graph_data.edge_attr,
                                        graph_data.batch)
        graph_feat = self.projection_head(self.pool(graph_feat, graph_data.batch))
        seq_data['amino_acid_indices'], seq_data['binding_indices'], seq_data['attention_mask'] = \
            seq_data['amino_acid_indices'].to(DEVICE),\
            seq_data['binding_indices'].to(DEVICE),\
            seq_data['attention_mask'].to(DEVICE)

        seq_feat = self.seq_encoder(seq_data['amino_acid_indices'],
                                    seq_data['binding_indices'],
                                    seq_data['attention_mask'])

        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            graph_feat_m = self.graph_encoder(graph_data.x,
                                              graph_data.edge_index,
                                              graph_data.edge_attr,
                                              graph_data.batch)
            graph_feat_m = self.projection_head(self.pool(graph_feat_m, graph_data.batch))
            graph_feat_all = torch.cat([graph_feat_m.t(), self.graph_queue.clone().detach()], dim=1)

            seq_feat_m = self.seq_encoder_m(seq_data['amino_acid_indices'],
                                            seq_data['binding_indices'],
                                            seq_data['attention_mask'])
            seq_feat_all = torch.cat([seq_feat_m.t(), self.seq_queue.clone().detach()], dim=1)

            sim_g2s_m = graph_feat_m @ seq_feat_all / self.temp
            sim_s2g_m = seq_feat_m @ graph_feat_all / self.temp

            sim_targets = torch.zeros(sim_g2s_m.size()).to(DEVICE)
            sim_targets.fill_diagonal_(1)

            sim_g2s_targets = alpha * F.softmax(sim_g2s_m, dim=1) + (1 - alpha) * sim_targets
            sim_s2g_targets = alpha * F.softmax(sim_s2g_m, dim=1) + (1 - alpha) * sim_targets

        sim_g2s = graph_feat @ seq_feat_all / self.temp
        sim_s2g = seq_feat @ graph_feat_all / self.temp

        loss_g2s = -torch.sum(F.log_softmax(sim_g2s, dim=1) * sim_g2s_targets, dim=1).mean()
        loss_s2g = -torch.sum(F.log_softmax(sim_s2g, dim=1) * sim_s2g_targets, dim=1).mean()

        loss_gsa = (loss_g2s + loss_s2g) / 2

        self._dequeue_and_enqueue(graph_feat_m, seq_feat_m)

        # ==============================================================================================
        # GSM Loss
        current_graph_feat, current_seq_feat = graph_feat, seq_feat
        for layer in self.bridge_tower:
            current_graph_feat, current_seq_feat = layer(current_graph_feat, current_seq_feat)
        output_pos = current_graph_feat + current_seq_feat

        with torch.no_grad():
            bs = graph_feat.size(0)
            weights_g2s = F.softmax(sim_g2s[:, :bs], dim=1)
            weights_s2g = F.softmax(sim_s2g[:, :bs], dim=1)

            weights_g2s.fill_diagonal_(0)
            weights_s2g.fill_diagonal_(0)

        # select a negative graph for each sequence
        graph_feat_neg = []
        for b in range(bs):
            neg_index = torch.multinomial(weights_g2s[b], 1).item()
            graph_feat_neg.append(graph_feat[neg_index])
        graph_feat_neg = torch.stack(graph_feat_neg, dim=0)

        # select a negative sequence for each graph
        seq_feat_neg = []
        for b in range(bs):
            neg_index = torch.multinomial(weights_s2g[b], 1).item()
            seq_feat_neg.append(seq_feat[neg_index])
        seq_feat_neg = torch.stack(seq_feat_neg, dim=0)

        seq_feat_all = torch.cat([seq_feat, seq_feat_neg], dim=0)
        graph_feat_all = torch.cat([graph_feat, graph_feat_neg], dim=0)

        current_graph_feat_all, current_seq_feat_all = graph_feat_all, seq_feat_all
        for layer in self.bridge_tower:
            current_graph_feat_all, current_seq_feat_all = layer(current_graph_feat_all, current_seq_feat_all)
        output_neg = current_graph_feat_all + current_seq_feat_all
        vl_embeddings = torch.cat([output_pos, output_neg], dim=0)
        vl_output = self.itm_head(vl_embeddings)

        gsm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2*bs, dtype=torch.long)],
                               dim=0).to(DEVICE)

        loss_gsm = F.cross_entropy(vl_output, gsm_labels)

        loss = loss_gsm + loss_gsa
        return loss

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)
                param_m.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, graph_feats, seq_feats):
        batch_size = graph_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0

        # Check if the last batch will exceed the queue size
        self.graph_queue[:, ptr:ptr + batch_size] = graph_feats.T
        self.seq_queue[:, ptr:ptr + batch_size] = seq_feats.T
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr
















































