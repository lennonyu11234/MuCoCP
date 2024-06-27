import json
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import selfies as sf


class DatasetLabeled(Dataset):
    def __init__(self, path):
        self.pep_ori = self.read_data(path)
        self.tokenizer = AutoTokenizer.from_pretrained("HELM-Triple")
        self.max_length_helm = 18
        with open('dataset/binding_vocab.json', 'r', encoding='utf-8') as f:
            self.binding_vocab = json.load(f)
        with open('dataset/amino_acid_vocab.json', 'r', encoding='utf-8') as f:
            self.amino_acid_vocab = json.load(f)

    def __len__(self):
        return len(self.pep_ori)

    def __getitem__(self, idx):
        HELM, label = self.pep_ori[idx]
        amino_acids, binding = self.extract_amino_acids(HELM)
        helm_indices = str(amino_acids + [binding])
        helm_index = self.tokenizer.encode(helm_indices)
        return helm_index, label

    def read_data(self, path):
        colomn_to_read = ['HELM',
                          'Permeability']
        pep = pd.read_csv(path, usecols=colomn_to_read)
        pep_ori = [(
            row['HELM'],
            row['Permeability']
        ) for _, row in pep.iterrows()]
        return pep_ori

    def extract_amino_acids(self, helm_notation):
        binding = ""

        start_index = helm_notation.find('{')
        end_index = helm_notation.find('}')
        if start_index != -1 and end_index != -1:
            sequence = helm_notation[start_index + 1:end_index]
            amino_acids = sequence.split('.')

        start_index_1 = helm_notation.rfind(',')
        end_index_1 = helm_notation.find('$', start_index_1)
        if start_index_1 != -1 and end_index_1 != -1 and start_index_1 != end_index_1:
            binding = helm_notation[start_index_1 + 1:end_index_1]
        return amino_acids, binding

    def padding(self, max_len, sequence):
        padded_sequence = sequence + [0] * (max_len - len(sequence))
        return padded_sequence

    def collate_fn(self, batch):
        sequences, helm_attention_mask, labels = [], [], []
        for sequence, label in batch:
            sequences.append(sequence)
            helm_attention_mask.append([1 for _ in range(len(sequence))] +
                                       [0 for _ in range(self.max_length_helm - len(sequence))])
            labels.append(label)
            # if label >= -6:
            #     labels.append(torch.tensor(1))
            # else:
            #     labels.append(torch.tensor(0))
            # if label >= -5.5:
            #     labels.append(torch.tensor(2))  # 类别1
            # elif -7 <= label < -5.5:
            #     labels.append(torch.tensor(1))  # 类别2
            # elif label < -7:
            #     labels.append(torch.tensor(0))  # 类别3

        padded_sequences = [self.padding(self.max_length_helm, seq) for seq in sequences]

        return {
            'helm_ids': torch.tensor(padded_sequences),
            'helm_mask': torch.tensor(helm_attention_mask),
            'label': torch.tensor(labels)
        }


class DatasetHELM(Dataset):
    def __init__(self, path):
        self.pep_ori = self.read_data(path)
        self.tokenizer = AutoTokenizer.from_pretrained("HELM-Triple")
        self.max_length_helm = 18
        with open('dataset/binding_vocab.json', 'r', encoding='utf-8') as f:
            self.binding_vocab = json.load(f)
        with open('dataset/amino_acid_vocab.json', 'r', encoding='utf-8') as f:
            self.amino_acid_vocab = json.load(f)

    def __len__(self):
        return len(self.pep_ori)

    def __getitem__(self, idx):
        HELM = self.pep_ori[idx]
        amino_acids, binding = self.extract_amino_acids(HELM)
        helm_indices = str(amino_acids + [binding])
        helm_index = self.tokenizer.encode(helm_indices)
        return helm_index

    def read_data(self, path):
        colomn_to_read = ['HELM']
        pep = pd.read_csv(path, usecols=colomn_to_read)

        pep_ori = [(row['HELM']) for _, row in pep.iterrows()]
        return pep_ori

    def extract_amino_acids(self, helm_notation):
        binding = ""

        start_index = helm_notation.find('{')
        end_index = helm_notation.find('}')
        if start_index != -1 and end_index != -1:
            sequence = helm_notation[start_index + 1:end_index]
            amino_acids = sequence.split('.')

        start_index_1 = helm_notation.rfind(',')
        end_index_1 = helm_notation.find('$', start_index_1)
        if start_index_1 != -1 and end_index_1 != -1 and start_index_1 != end_index_1:
            binding = helm_notation[start_index_1 + 1:end_index_1]
        return amino_acids, binding

    def padding(self, max_len, sequence):
        padded_sequence = sequence + [0] * (max_len - len(sequence))
        return padded_sequence

    def collate_fn(self, batch):
        sequences, helm_attention_mask = [], []
        for sequence in batch:
            sequences.append(sequence)
            helm_attention_mask.append([1 for _ in range(len(sequence))] +
                                       [0 for _ in range(self.max_length_helm - len(sequence))])

        padded_sequences = [self.padding(self.max_length_helm, seq) for seq in sequences]

        return {
            'helm_ids': torch.tensor(padded_sequences),
            'helm_mask': torch.tensor(helm_attention_mask)
        }


if __name__ == "__main__":
    dataset = DatasetLabeled(path='dataset/for train/pretrain.csv')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=dataset.collate_fn)
    for count, i in enumerate(dataloader):
        print(i['helm_ids'])
        print(i['helm_mask']),
        print(i['label'])
        if count > 10:
            break


