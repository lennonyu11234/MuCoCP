import torch
import pandas as pd
from torch.utils.data import Dataset

class DatasetCyclic(Dataset):
    def __init__(self, path):
        super().__init__()
        self.pep_ori = self.read_data(path=path)
        self.amino_acid_vocab = self.build_acid_vocab()
        self.binding_vocab = self.build_binding_vocab()
        self.max_len = self.find_max_sequence_length()

    def find_max_sequence_length(self):
        max_length = 0
        for data_point in self.pep_ori:
            HELM = data_point[0]
            amino_acids, _ = self.extract_amino_acids(HELM)
            sequence_length = len(amino_acids)
            if sequence_length > max_length:
                max_length = sequence_length
        return max_length

    def __len__(self):
        return len(self.pep_ori)

    def __getitem__(self, index):
        HELM, LogP, TPSA, label = self.pep_ori[index]
        amino_acids, binding = self.extract_amino_acids(HELM)
        amino_acid_indices, binding_index = self.index_conversion(amino_acids, binding)

        return amino_acid_indices, binding_index, TPSA, LogP, label

    def read_data(self, path):
        columns_to_read = ['CycPeptMPDB_ID', 'HELM', 'Permeability', 'Sequence_LogP', 'Sequence_TPSA']
        pep = pd.read_csv(filepath_or_buffer=path, usecols=columns_to_read)
        pred_pep = [
            (row['HELM'], row['Sequence_LogP'],
             row['Sequence_TPSA'], row['Permeability']) for _, row in pep.iterrows()]
        return pred_pep

    def collate_fn(self, batch):
        sequences, bindings, TPSA, LogP, labels, attention_mask = [], [], [], [], [], []
        for sequence, binding, tpsa, logp, label in batch:
            sequences.append(sequence)
            bindings.append(binding)
            TPSA.append(tpsa)
            LogP.append(logp)
            labels.append(label)
            attention_mask.append([True for _ in range(len(sequence))] +
                                  [False for _ in range(self.max_len-len(sequence))])

        padded_sequences = [self.padding(self.max_len, seq) for seq in sequences]

        return {
            'amino_acid_indices': torch.tensor(padded_sequences),
            'binding_indices': torch.tensor(bindings),
            # 'TPSA': torch.tensor(TPSA),
            # 'LogP': torch.tensor(LogP),
            'labels': torch.tensor(labels),
            'attention_mask': torch.tensor(attention_mask)
        }

    def padding(self, max_len, sequence):
        # print(self.max_len)
        padded_sequence = sequence + [0] * (max_len - len(sequence))
        return padded_sequence

    def build_acid_vocab(self):
        amino_acid_vocab = set()
        for data_point in self.pep_ori:
            HELM = data_point[0]
            amino_acids, _ = self.extract_amino_acids(HELM)
            amino_acid_vocab.update(amino_acids)
        amino_acid_vocab = {aa: idx + 1 for idx, aa in enumerate(sorted(amino_acid_vocab))}
        return amino_acid_vocab

    def build_binding_vocab(self):
        binding_vocab = set()
        for data_point in self.pep_ori:
            HELM = data_point[0]
            _, binding = self.extract_amino_acids(HELM)
            binding_vocab.add(binding)
        binding_vocab = {bind: idx + 1 for idx, bind in enumerate(sorted(binding_vocab))}
        return binding_vocab

    def index_conversion(self, amino_acids, binding):
        amino_acid_indices = [self.amino_acid_vocab[aa] for aa in amino_acids]
        binding_index = self.binding_vocab[binding] if binding in self.binding_vocab else -1
        return amino_acid_indices, binding_index

    def extract_amino_acids(self, helm_notation):
        amino_acids = set()
        binding = ""

        start_index = helm_notation.find('{')
        end_index = helm_notation.find('}')
        if start_index != -1 and end_index != -1:
            sequence = helm_notation[start_index + 1:end_index]
            amino_acids.update(sequence.split('.'))

        start_index_1 = helm_notation.find(',', helm_notation.find(',') + 1)
        end_index_1 = helm_notation.find('$', start_index_1)
        if start_index_1 != -1 and end_index_1 != -1 and start_index_1 != end_index_1:
            binding = helm_notation[start_index_1 + 1:end_index_1]

        return amino_acids, binding

    def get_amino_acid_vocab_size(self):
        return len(self.amino_acid_vocab)

    def get_binding_vocab_size(self):
        return len(self.binding_vocab)