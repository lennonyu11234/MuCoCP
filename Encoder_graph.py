import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot, zeros

num_atom_type = 35
num_chirality_tag = 3

num_bond_type = 6
num_bond_direction = 3

# ==========================================================================================
class GraphSAGE(MessagePassing):
    def __init__(self, emb_dim, aggr="mean"):
        super().__init__()
        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)
        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # print(edge_attr.shape, edge_attr.dtype, edge_attr.device)
        # print(x.shape)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4
        # print(self_loop_attr.shape, self_loop_attr.dtype, self_loop_attr.device)
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        # print(self_loop_attr.shape, self_loop_attr.dtype, self_loop_attr.device)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        # print(edge_attr.shape, edge_attr.dtype, edge_attr.device)
        edge_embedding = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        print(edge_index.shape)
        print(x.shape)
        print(edge_embedding.shape)

        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_embedding, aggr=self.aggr)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)


class GINLayer(MessagePassing):
    def __init__(self, emb_dim, aggr='add'):
        super().__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(2*emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)
        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4
        # print(self_loop_attr.shape, self_loop_attr.dtype, self_loop_attr.device)
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        # print(self_loop_attr.shape, self_loop_attr.dtype, self_loop_attr.device)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        # print(edge_attr.shape, edge_attr.dtype, edge_attr.device)
        edge_embedding = self.edge_embedding1(edge_attr[:, 0]) + \
                         self.edge_embedding2(edge_attr[:, 1])

        return self.propagate(edge_index=edge_index, x=x,
                              edge_attr=edge_embedding, aggr=self.aggr)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNLayer(MessagePassing):
    def __init__(self, emb_dim, aggr='add'):
        super().__init__()
        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type+1, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction+1, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        # Calculate normalization directly without using scatter_add
        row, _ = edge_index
        num_nodes = x.size(0)
        dtype = x.dtype
        deg = torch.bincount(row, minlength=num_nodes).to(dtype=dtype, device=x.device)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[row]

        x = self.linear(x)

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings, norm=norm,
                              aggr=self.aggr)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


# ==========================================================================================
class GNN(torch.nn.Module):
    def __init__(self, num_layer,
                 emb_dim,
                 JK="last",
                 drop_ratio=0,
                 gnn_type="graphsage"):
        super().__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)
        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.gnn = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == 'graphsage':
                self.gnn.append(GraphSAGE(emb_dim))
            elif gnn_type == 'gin':
                self.gnn.append(GINLayer(emb_dim))
            elif gnn_type == 'gcn':
                self.gnn.append(GCNLayer(emb_dim))

        self.batch_norm = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norm.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, x, edge_index, edge_attr):

        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnn[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norm[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        return node_representation


class GNN_graphpred(torch.nn.Module):
    def __init__(self, num_layer,
                 emb_dim,
                 JK='last',
                 drop_ratio=0,
                 graph_pooling='mean',
                 gnn_type='graphsage'):
        super().__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type)

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

            # For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1

        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, 1)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, 1)

    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file))

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)

        return self.graph_pred_linear(self.pool(node_representation, batch))


class GNNEncoder(torch.nn.Module):
    def __init__(self, num_layer,
                 emb_dim,
                 JK='last',
                 drop_ratio=0,
                 graph_pooling='mean',
                 gnn_type='graphsage'):
        super().__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type)
        # self.pool = global_mean_pool
        # self.projection_head = torch.nn.Sequential(torch.nn.Linear(512, 512),
        #                                            torch.nn.ReLU(inplace=True),
        #                                            torch.nn.Linear(512, 512))

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

            # For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1

        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, 1)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, 1)

        # for param in self.gnn.parameters():
        #     param.requires_grad = False

    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file))

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)
        # node_representation = self.pool(node_representation, batch)
        # node_representation = self.projection_head(node_representation)

        return node_representation