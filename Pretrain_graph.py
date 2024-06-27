import argparse
import os
from Dataset_graph import CyclicPepDatasetDropN, CyclicPepDatasetMaskN
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from Encoder_graph import GNN


class GraphCL(nn.Module):
    def __init__(self, gnn):
        super().__init__()
        self.gnn = gnn
        self.pool = global_mean_pool
        self.projection_head = nn.Sequential(nn.Linear(512, 512),
                                             nn.ReLU(inplace=True),
                                             nn.Linear(512, 512))

    def forward_cl(self, x, edge_index, edge_attr, batch):
        x = self.gnn(x, edge_index, edge_attr)
        x = self.pool(x, batch)
        x = self.projection_head(x)
        return x

    def loss_cl(self, x1, x2):
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        a1 = torch.einsum('ik,jk->ij', x1, x2)
        a2 = torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix_ = a1 / a2
        sim_matrix = torch.exp(sim_matrix_ / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        all_sim = sim_matrix.sum(dim=1)
        neg_sim = all_sim - pos_sim
        loss = pos_sim / neg_sim
        loss = - torch.log(loss).mean()
        # if loss < 0:
        #     print(1)

        return loss


def train(args, model, device, loader1, loader2, optimizer, epoch, epochs):

    train_acc_accum = 0
    train_loss_accum = 0
    train_loop = tqdm(zip(loader1, loader2), total=len(loader2), desc='Train', colour='white')
    for step, batch in enumerate(train_loop):
        batch1, batch2 = batch
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)

        optimizer.zero_grad()
        # print("batch", batch1.x.shape, batch2.x.shape)

        x1 = model.forward_cl(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch)
        x2 = model.forward_cl(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch)
        loss = model.loss_cl(x1, x2)

        loss.backward()
        optimizer.step()

        train_loss_accum += float(loss.detach().cpu().item())
        acc = torch.tensor(0)
        train_acc_accum += float(acc.detach().cpu().item())

        train_loop.set_description(f'Epoch [{epoch+1}/{epochs}]')
        train_loop.set_postfix(loss=train_loss_accum/(step+1))

    return train_acc_accum / (step + 1), train_loss_accum / (step + 1)


def main():
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN_CYC_baseline message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=512,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.1,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gcn")
    parser.add_argument('--seed', type=int, default=3407, help="Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataset loading')
    args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # set up dataset
    root_cyc_dropn = 'dataset/graph/data_dropN/'
    dataset_cyc_dropn = CyclicPepDatasetDropN(root=root_cyc_dropn)

    root_cyc_MaskN = 'dataset/graph/data_MaskN/'
    dataset_cyc_MaskN = CyclicPepDatasetMaskN(root=root_cyc_MaskN)

    # set up model
    gnn = GNN(args.num_layer, args.emb_dim,
              JK=args.JK,
              drop_ratio=args.dropout_ratio,
              gnn_type=args.gnn_type)
    model = GraphCL(gnn)
    model.to(device)
    print(model)
    # for name, param in model.named_parameters():
    #     print(name, param.shape)

    # set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    loader1 = DataLoader(dataset_cyc_dropn,
                         batch_size=args.batch_size,
                         num_workers=args.num_workers,
                         shuffle=False)
    loader2 = DataLoader(dataset_cyc_MaskN,
                         batch_size=args.batch_size,
                         num_workers=args.num_workers,
                         shuffle=False)
    model.train()
    for epoch in range(1, args.epochs + 1):

        # import time
        # past = time.time()

        train_acc, train_loss = train(args, model, device, loader1, loader2, optimizer, epoch, epochs=args.epochs)

        # now = time.time()
        # print(now-past)

        # print("Train Accuracy::", train_acc)
        # print("Train Loss:", train_loss)
        save_dir = "model/Graph_Pretrain/Drop_Nodes/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if epoch == 20:
            torch.save(model.gnn.state_dict(), save_dir + 'DropN&MaskN' + ".pth")

        # csv_filename = 'D:\dataset\experiment_result\pretrain_loss_record.csv'
        # with open(csv_filename, mode='a' if os.path.exists(csv_filename) else 'w', newline='') as file:
        #     writer = csv.writer(file)
        #     if os.path.getsize(csv_filename) == 0:
        #         writer.writerow(['Epoch', 'DropNNonC_MaskN'])
        #     writer.writerow([epoch, f"{train_loss:.3f}"])


if __name__ == "__main__":
    main()


























