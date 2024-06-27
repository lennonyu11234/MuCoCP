import argparse
import os
from torch_geometric.loader import DataLoader as DataLoaderGraph
from torch.utils.data import DataLoader as DataLoaderSeq
from Dataset_graph import CyclicPepDataset, CyclicPepDatasetAddHydrogenBonds
from Dataset_seq import DatasetCyclic as DatasetSeq
import torch.utils.checkpoint
from torch import optim
from Pretrain_Multimodal_module import PretrainMultimodal
from tqdm import tqdm
from sklearn.utils import shuffle as sk_shuffle
from torch.utils.data import Subset


# =====================================================[parser]==============================================
parser = argparse.ArgumentParser(description='ALBEF Multimodal Pretrain')
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.03)')
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


args = parser.parse_args()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================================[set up dataset]=================================================
root_cyc = 'dataset/graph/data/'
root_cyc_add_hydrogen_bonds = 'dataset/graph/data_add_hydrogen_bonds/'
path = 'dataset/CycPeptMPDB_Peptide_ALL.csv'


# graph dataset
dataset_original = CyclicPepDataset(root=root_cyc)
dataset_cyc_add_hydrogen_bonds = CyclicPepDatasetAddHydrogenBonds(root=root_cyc_add_hydrogen_bonds)

dataset_graph = dataset_original + dataset_cyc_add_hydrogen_bonds
print(len(dataset_graph))
# sequence dataset
dataset_sequence = DatasetSeq(path)
dataset_seq = dataset_sequence + dataset_sequence
print(len(dataset_seq))

# 打乱排列
num_samples = len(dataset_graph)
indices = list(range(num_samples))
shuffled_indices = sk_shuffle(indices)

# 使用 Subset 创建随机排列的子集
subset_graph = Subset(dataset_graph, shuffled_indices)
subset_seq = Subset(dataset_seq, shuffled_indices)

# ========================================[set up dataloader]===============================================
loader_graph = DataLoaderGraph(subset_graph,
                               batch_size=args.batch_size,
                               shuffle=False, drop_last=True)
loader_seq = DataLoaderSeq(subset_seq,
                           batch_size=args.batch_size,
                           collate_fn=dataset_sequence.collate_fn,
                           shuffle=False, drop_last=True)

# ==========================================[set up module]=================================================
model = PretrainMultimodal(num_blocks=args.num_blocks)
model.to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=args.lr)


# =======================================[set up train function]=============================================
def train(model, device, loader_graph, loader_seq, optimizer, epoch, epochs):
    train_loss_accum = 0.0
    train_loop = tqdm(zip(loader_graph, loader_seq), total=len(loader_seq), desc='pretrain', colour='white')
    for step, batch in enumerate(train_loop):
        graph_batch, seq_batch = batch
        graph_batch = graph_batch.to(device)
        optimizer.zero_grad()

        loss = model(graph_batch, seq_batch)
        loss.backward()
        optimizer.step()

        train_loss_accum += float(loss.detach().cpu().item())
        train_loop.set_description(f'Epoch [{epoch + 1}/{epochs}]')
        train_loop.set_postfix(loss=train_loss_accum / (step + 1))


# ==================================================[train]==================================================
for epoch in range(1, args.epochs+1):
    train(model, device=DEVICE,
          loader_graph=loader_graph, loader_seq=loader_seq,
          optimizer=optimizer, epoch=epoch, epochs=args.epochs)
    save_dir = "model/Multimodal_Pretrain/AddHB&AddHB/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if epoch % 10 == 0:
        torch.save(model.graph_encoder.gnn.state_dict(), save_dir + "Pretrain_graph_enc" + ".pth")
        torch.save(model.seq_encoder.state_dict(), save_dir + "Pretrain_seq_enc" + ".pth")
        # torch.save(model.bridge_tower.state_dict(), save_dir + "Pretrain_bridge" + ".pth")
