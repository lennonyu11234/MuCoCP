import argparse
import os
from torch_geometric.loader import DataLoader as DataLoaderGraph
from torch.utils.data import DataLoader as DataLoaderSeq
from Encoder_seq import SeqEncoder
from Encoder_graph import GNNEncoder
from Encoder_Multimodal import BridgeTowerEncoder
from Dataset_graph import CyclicPepDataset as DatasetGraph
from Dataset_seq import DatasetCyclic as DatasetSeq
import torch
import torch.utils.checkpoint
from torch import nn, optim
import math
import csv
import numpy as np
from sklearn.metrics import mean_absolute_error, explained_variance_score, r2_score, mean_squared_error


# =====================================[parser]==============================================
criterion_cla = nn.BCEWithLogitsLoss()
criterion_reg = nn.MSELoss()
parser = argparse.ArgumentParser(description='Train and Validation Function')
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
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


# =====================================[utils]=================================================
def write_to_csv(epoch, metrics):
    with open('', mode='a', newline='') as file:
        writer = csv.writer(file)
        if epoch == 1:
            writer.writerow(['Epoch', 'Sensitivity', 'Specificity', 'Precision', 'Accuracy', 'F1_Score', 'MCC'])
        writer.writerow([epoch] + metrics)


def calculate_metrics(conf_matrix):
    TP = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    TN = conf_matrix[1][1]
    FN = conf_matrix[1][0]

    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    F1_score = 2 * ((precision * sensitivity) / (precision + sensitivity)) if (precision + sensitivity) != 0 else 0
    mcc_numerator = (TP * TN) - (FP * FN)
    mcc_denominator = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) if ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) != 0 else 1
    mcc = mcc_numerator / mcc_denominator

    return {
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "Accuracy": accuracy,
        "F1_Score": F1_score,
        "MCC": mcc
    }


# ====================================[set dataloader]==========================================
root_train = 'D:/dataset/data_train/'
root_test = 'D:/dataset/data_test/'
path_train = 'D:/dataset/dataset_2000_train.csv'
path_test = 'D:/dataset/dataset_2000_test.csv'
dataset_train_seq = DatasetSeq(path_train)
dataset_test_seq = DatasetSeq(path_test)

dataset_train_graph = DatasetGraph(root_train)
dataset_test_graph = DatasetGraph(root_test)

train_loader_graph = DataLoaderGraph(dataset_train_graph,
                                     batch_size=32, shuffle=False)
train_loader_seq = DataLoaderSeq(dataset_train_seq, collate_fn=dataset_train_seq.collate_fn,
                                 batch_size=32, shuffle=False)

test_loader_graph = DataLoaderGraph(dataset_test_graph,
                                    batch_size=32, shuffle=False)
test_loader_seq = DataLoaderSeq(dataset_test_seq, collate_fn=dataset_test_seq.collate_fn,
                                batch_size=32, shuffle=False)


# ===================================[Train_Test_Function]============================================
def train_classification(model, device,
                         train_loader_graph, train_loader_seq,
                         optimizer, epoch):
    # Train part
    train_loss_accum = 0
    train_loop_train = zip(train_loader_graph, train_loader_seq)
    for step, batch in enumerate(train_loop_train):
        batch1, batch2 = batch
        batch1 = batch1.to(device)
        label = batch1.label
        label = torch.where(label > -6.0, torch.tensor(1.0), torch.tensor(0.0))

        optimizer.zero_grad()
        output, _ = model.forward(batch1, batch2)
        label = label.view(output.shape)
        loss = criterion_cla(output, label)
        loss.backward()
        optimizer.step()
        train_loss_accum += float(loss.detach().cpu().item())


def train_regression(model, device,
                     train_loader_graph, train_loader_seq,
                     optimizer, epoch):
    train_loss_accum = 0
    train_loop = zip(train_loader_graph, train_loader_seq)
    for step, batch in enumerate(train_loop):
        batch_graph, batch_seq = batch
        batch_graph = batch_graph.to(device)
        label = batch_graph.label

        optimizer.zero_grad()
        output, _ = model.forward(batch_graph, batch_seq)
        label = label.view(output.shape)
        loss = criterion_reg(output, label)
        loss.backward()
        optimizer.step()
        train_loss_accum += float(loss.detach().cpu().item())


def test_classification(model, device,
                        test_loader_graph, test_loader_seq):
    model.eval()
    TP = 0  # True Positives
    FP = 0  # False Positives
    TN = 0  # True Negatives
    FN = 0  # False Negatives

    with torch.no_grad():
        test_loop = zip(test_loader_graph, test_loader_seq)
        for step, batch in enumerate(test_loop):
            batch1, batch2 = batch
            batch1 = batch1.to(device)

            amino_acid = batch2['amino_acid_indices'].clone().detach()
            binding = batch2['binding_indices'].clone().detach()
            mask = batch2['attention_mask'].clone().detach()
            label = batch2['labels'].clone().detach()

            amino_acid, binding, mask, label = amino_acid.to(device), binding.to(device), \
                mask.to(device), label.to(device)
            label = torch.where(label >= -6, torch.tensor(1.0), torch.tensor(0.0))

            pred, _ = model.forward(batch1, batch2)

            for idx in range(label.size(0)):
                sample_pred = pred[idx]
                sample_label = label[idx]

                if sample_label == 1 and sample_pred >= 0.5:
                    TP += 1
                if sample_label == 1 and sample_pred < 0.5:
                    FP += 1
                if sample_label == 0 and sample_pred < 0.5:
                    TN += 1
                if sample_label == 0 and sample_pred >= 0.5:
                    FN += 1

    confusion_matrix = [[TP, FN], [FP, TN]]
    metrics = calculate_metrics(confusion_matrix)

    accuracy = metrics["Accuracy"]
    f1_score = metrics["F1_Score"]
    mcc = metrics["MCC"]
    sensitivity = metrics["Sensitivity"]
    specificity = metrics["Specificity"]

    write_to_csv(metrics)

    print('==== Testing Evaluation ====')
    print(f'Accuracy: {accuracy:.3f}')
    print(f'F1 Score: {f1_score:.3f}')
    print(f'MCC: {mcc:.3f}')
    print(f'Sensitivity: {sensitivity:.3f}')
    print(f'Specificity: {specificity:.3f}')


def test_regression(model, device,
                    test_loader_graph, test_loader_seq):
    model.eval()
    with torch.no_grad():
        preds, labels = [], []
        test_loop = zip(test_loader_graph, test_loader_seq)
        for step, batch in enumerate(test_loop):
            batch_graph, batch_seq = batch
            batch_graph = batch_graph.to(device)
            label = batch_graph.label

            pred, _ = model.forward(batch_graph, batch_seq)
            label = label.view(pred.shape)
            print(label, pred)

            preds.append(pred.cpu().numpy())
            labels.append(label.cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(labels, axis=0)

        r2_Score = r2_score(labels, preds)
        MSE = mean_squared_error(labels, preds)
        RMSE = np.sqrt(mean_squared_error(labels, preds))
        MAE = mean_absolute_error(labels, preds)
        Explained_variance_score = explained_variance_score(labels, preds)

        write_to_csv(r2_Score, MSE, RMSE, MAE, Explained_variance_score)

        print(f"==========[Test]==========")
        print(f'===[r2_Score: {r2_Score:.3f}]===')
        print(f'===[MSE: {MSE:.3f}]===')
        print(f'===[MAE: {MAE:.3f}]===')
        print(f'===[RMSE: {RMSE:.3f}]===')
        print(f'===[Explained_variance_score: {Explained_variance_score:.3f}]===\n')


# ====================================[set encoder]==========================================
seq_encoder = SeqEncoder(seq_vocab=args.amino_vocab_size,
                         bind_vocab=args.binding_vocab_size).to(DEVICE)
graph_encoder = GNNEncoder(num_layer=args.graph_num_layers, emb_dim=args.emb_dim,
                           drop_ratio=args.dropout_ratio,
                           JK=args.JK, gnn_type=args.gnn_type).to(DEVICE)

# ===================================[set model]============================================
model_file_seq = 'model/Multimodal_pretrain/AddHB&AddHB/Pretrain_graph_enc.pth'
model_file_graph = 'model/Multimodal_pretrain/AddHB&AddHB/Pretrain_seq_enc.pth'

model = BridgeTowerEncoder(seq_enc=seq_encoder, graph_enc=graph_encoder, num_blocks=args.num_blocks)
model.from_pretrained(model_file_seq=model_file_seq,
                      model_file_graph=model_file_graph)
model.to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=args.lr)

# ======================================[train]=============================================
for epoch in range(1, args.epochs+1):
    # classification
    train_classification(model=model, device=DEVICE,
                         train_loader_graph=dataset_train_graph,
                         train_loader_seq=dataset_train_seq,
                         optimizer=optimizer,
                         epoch=epoch)
    save_dir = "new_stage/best_model_dataset1/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = save_dir + 'best_model_frozen.pth'
    if epoch == args.epochs:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f'Removing existing model file')
        torch.save(model.state_dict(), file_path)


# ====================================[set test model]======================================
model = BridgeTowerEncoder(seq_enc=seq_encoder, graph_enc=graph_encoder, num_blocks=args.num_blocks)
model_file = 'model/best_model_dataset1/best_model_classification/Multimodal_pretrain/AddHB&AddHB/best_model_frozen.pth'
model.load_state_dict(torch.load(model_file))
model.to(DEVICE)


# ==========================================[test]==========================================
test_regression(model,
                device=DEVICE,
                test_loader_graph=test_loader_graph,
                test_loader_seq=test_loader_seq)





