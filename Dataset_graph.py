import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
from torch_geometric.loader import DataLoader as DataLoaderGraph


allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)),
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list': [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs': [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT]}

# ==================================utils=====================================


def normalize_data(data):
    min_val = min(data)
    max_val = max(data)
    normalized = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized


def read_data_cyclic(path):
    columns_to_read = ['CycPeptMPDB_ID', 'SMILES', 'Permeability']
    pep = pd.read_csv(filepath_or_buffer=path, usecols=columns_to_read)
    pred_pep = [(row['CycPeptMPDB_ID'], row['SMILES'], row['Permeability']) for index, row in pep.iterrows()]
    return pred_pep


data_cyc = read_data_cyclic('')


# ==================================augmentation=============================
def mol_to_graph_data(mol):
    # atom
    num_atom_feature = 2
    atom_feature_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
                           atom.GetAtomicNum())] + \
                       [allowable_features['possible_chirality_list'].index(
                           atom.GetChiralTag())]
        atom_feature_list.append(atom_feature)
    x = torch.tensor(np.array(atom_feature_list), dtype=torch.long)

    # bond
    num_bond_features = 2
    if len(mol.GetBonds()) > 0:
        edges_list = []
        edges_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                               bond.GetBondType())] + \
                           [allowable_features['possible_bond_dirs'].index(
                               bond.GetBondDir())]
            edges_list.append((i, j))
            edges_features_list.append(edge_feature)
            edges_list.append((j, i))
            edges_features_list.append(edge_feature)

        # data.edge_index
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr
        edge_attr = torch.tensor(np.array(edges_features_list), dtype=torch.long)

    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return graph_data


def mol_to_graph_data_noneH(mol):
    # atom
    num_atom_feature = 2
    atom_feature_list = []
    atom_idx_map = {}  # Mapping atom indices from original molecule to reduced molecule
    reduced_atom_idx = 0
    non_hydrogen_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1]

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:  # Exclude hydrogen atoms
            atom_feature = [allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum())] + \
                           [allowable_features['possible_chirality_list'].index(atom.GetChiralTag())]
            atom_feature_list.append(atom_feature)
            atom_idx_map[atom.GetIdx()] = reduced_atom_idx
            reduced_atom_idx += 1

    x = torch.tensor(np.array(atom_feature_list), dtype=torch.long)

    # bond
    num_bond_features = 2
    if len(mol.GetBonds()) > 0:
        edges_list = []
        edges_features_list = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            if i in non_hydrogen_atoms and j in non_hydrogen_atoms:  # Check if both atoms are non-hydrogen
                if mol.GetAtomWithIdx(i).GetAtomicNum() != 1 and mol.GetAtomWithIdx(j).GetAtomicNum() != 1:
                    i_reduced = atom_idx_map[i]
                    j_reduced = atom_idx_map[j]

                    edge_feature = [allowable_features['possible_bonds'].index(bond.GetBondType())] + \
                                   [allowable_features['possible_bond_dirs'].index(bond.GetBondDir())]

                    edges_list.append((i_reduced, j_reduced))
                    edges_features_list.append(edge_feature)
                    edges_list.append((j_reduced, i_reduced))
                    edges_features_list.append(edge_feature)

        # data.edge_index
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr
        edge_attr = torch.tensor(np.array(edges_features_list), dtype=torch.long)

    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return graph_data


def drop_nodes(data):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()

    drop_num = int(node_num * 0.2)
    idx_perm = np.random.permutation(node_num)
    idx_drop = idx_perm[:drop_num]
    idx_nondrop = idx_perm[drop_num:]
    idx_nondrop.sort()
    idx_dict = {idx_nondrop[n]: n for n in list(range(idx_nondrop.shape[0]))}

    edge_index = data.edge_index.numpy()
    edge_mask = np.array([n for n in range(edge_num) if
                          not (edge_index[0, n] in idx_drop or
                               edge_index[1, n] in idx_drop)])

    edge_index = [[idx_dict[edge_index[0, n]],
                   idx_dict[edge_index[1, n]]] for n
                  in range(edge_num) if(not edge_index[0, n] in idx_drop) and
                                       (not edge_index[1, n] in idx_drop)]
    try:
        data.edge_index = torch.tensor(edge_index).transpose_(0, 1)
        data.x = data.x[idx_nondrop]
        data.edge_attr = data.edge_attr[edge_mask]
    except:
        data = data

    return data


def drop_nodes_nonC(data):
    # print('data.x:', data.x)
    # print('data.edge_index:', data.edge_index)
    # print('data.edge_attr:', data.edge_attr)
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num * 0.2)

    atom_number = data.x[:, 0].numpy()
    c_idx = np.where(atom_number == 5)[0]
    nonc_idx = np.where(atom_number != 5)[0]

    # get drop_idx
    if drop_num <= len(nonc_idx):
        idx_drop = np.random.choice(nonc_idx, drop_num, replace=False)
    else:
        tmp = np.random.choice(c_idx, drop_num - len(nonc_idx), replace=False)
        idx_drop = np.concatenate([nonc_idx, tmp], axis=0)

    idx_nondrop = [n for n in range(node_num) if not n in idx_drop]

    # print('idx_drop,', idx_drop)
    # print('idx_nondrop', idx_nondrop)

    # drop node features
    ## data.x = data.x[idx_nondrop]

    # modify edge index and feature
    edge_index = data.edge_index.numpy()
    idx_drop = set(idx_drop)
    idx_nondrop_edge = []
    for i in range(edge_index.shape[1]):
        tmp = set(edge_index[:, i])
        if not idx_drop.intersection(tmp):
            idx_nondrop_edge.append(i)

    edge_index = torch.from_numpy(edge_index[:, idx_nondrop_edge])
    edge_attr = data.edge_attr[idx_nondrop_edge, :]

    data.edge_index = edge_index
    data.edge_attr = edge_attr
    if data.edge_index.shape[1] != data.edge_attr.shape[0]:
        print('data dropping failed!')
        return
        # print(data.x)
    # print(data.edge_index)
    # print(data.edge_attr)

    return data


def mol_to_graph_data_maskn(mol):
    # atom
    num_atom_feature = 2
    atom_feature_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
                           atom.GetAtomicNum())] + \
                       [allowable_features['possible_chirality_list'].index(
                           atom.GetChiralTag())]
        atom_feature_list.append(atom_feature)
    x = torch.tensor(np.array(atom_feature_list), dtype=torch.long)

    # bond
    num_bond_features = 2
    if len(mol.GetBonds()) > 0:
        edges_list = []
        edges_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                               bond.GetBondType())] + \
                           [allowable_features['possible_bond_dirs'].index(
                               bond.GetBondDir())]
            edges_list.append((i, j))
            edges_features_list.append(edge_feature)
            edges_list.append((j, i))
            edges_features_list.append(edge_feature)

        # data.edge_index
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr
        edge_attr = torch.tensor(np.array(edges_features_list), dtype=torch.long)

    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    node_num = graph_data.num_nodes

    mask_num = int(node_num * 0.2)

    x = graph_data.x.to(torch.float)
    token = x.mean(dim=0)
    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    graph_data.x[idx_mask] = torch.tensor(token, dtype=torch.long)

    return graph_data


def mol_to_graph_data_add__hydrogen_bonds(mol):
    num_atom_feature = 2
    atom_feature_list = []
    carbonyl_oxygen_list = []
    nitrogen_hydrogen_list = []
    hydrogen_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 1]  # Hydrogen atoms

    for atom in mol.GetAtoms():
        atom_feature = [
            allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum()),
            allowable_features['possible_chirality_list'].index(atom.GetChiralTag())
        ]
        atom_feature_list.append(atom_feature)

    x = torch.tensor(np.array(atom_feature_list), dtype=torch.long)

    # bond
    num_bond_features = 2
    if len(mol.GetBonds()) > 0:
        edges_list = []
        edges_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [
                allowable_features['possible_bonds'].index(bond.GetBondType()),
                allowable_features['possible_bond_dirs'].index(bond.GetBondDir())
            ]
            edges_list.append((i, j))
            edges_features_list.append(edge_feature)
            edges_list.append((j, i))
            edges_features_list.append(edge_feature)

        # Find carbonyl oxygens (assuming C=O is the carbonyl functional group)
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                if bond.GetBeginAtom().GetAtomicNum() == 6 and bond.GetEndAtom().GetAtomicNum() == 8:
                    carbonyl_oxygen_list.append(bond.GetEndAtomIdx())

        # Create hydrogen bonds between carbonyl oxygens and hydrogen atoms
        for oxygen_idx in carbonyl_oxygen_list:
            if len(hydrogen_atoms) > 0:
                hydrogen_idx = random.choice(hydrogen_atoms)  # Randomly select a hydrogen atom
                edges_list.append((oxygen_idx, hydrogen_idx))
                edges_features_list.append(
                    [len(allowable_features['possible_bonds']),
                     len(allowable_features['possible_bond_dirs'])])  # Extra bond type for hydrogen bond
                hydrogen_atoms.remove(hydrogen_idx)  # Remove the selected hydrogen atom from the list
        # data.edge_index
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr
        edge_attr = torch.tensor(np.array(edges_features_list), dtype=torch.long)

    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return graph_data


# ===============================dataset=====================================
class CyclicPepDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None):
        super(CyclicPepDataset, self).__init__(root, transform, pre_transform)  # transform就是数据增强，对每一个数据都执行
        self.root = root
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['processed_cyc_data.dataset']

    def download(self):
        pass

    def process(self):
        graph_data_list = []

        # Assuming `data_cyc` is a list of tuples containing (id, pep_smile, Permeability)
        Permeability_values = [item[2] for item in tqdm(data_cyc, desc='Processing peptides')]
        # normalized_values = normalize_data(Permeability_values)

        for idx, (id, pep_smile, permeability) in enumerate(tqdm(data_cyc,
                                                                 desc='Processing peptides')):
            molecule = Chem.MolFromSmiles(pep_smile)
            molecule = Chem.AddHs(molecule)
            AllChem.EmbedMolecule(molecule)
            data = mol_to_graph_data(molecule)
            data.label = torch.tensor(Permeability_values[idx], dtype=torch.float)

            graph_data_list.append(data)

        data, slice = self.collate(graph_data_list)
        torch.save((data, slice), self.processed_paths[0])


class CyclicPepDatasetNoneH(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None):
        super(CyclicPepDatasetNoneH, self).__init__(root, transform, pre_transform)  # transform就是数据增强，对每一个数据都执行
        self.root = root
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['processed_cyc_data.dataset']

    def download(self):
        pass

    def process(self):
        graph_data_list = []

        # Assuming `data_cyc` is a list of tuples containing (id, pep_smile, Permeability)
        Permeability_values = [item[2] for item in tqdm(data_cyc, desc='Processing peptides')]
        normalized_values = normalize_data(Permeability_values)

        for idx, (id, pep_smile, permeability) in enumerate(tqdm(data_cyc, desc='Processing peptides')):
            molecule = Chem.MolFromSmiles(pep_smile)
            molecule = Chem.AddHs(molecule)
            AllChem.EmbedMolecule(molecule)
            data = mol_to_graph_data_noneH(molecule)
            data.label = torch.tensor(normalized_values[idx], dtype=torch.float)

            graph_data_list.append(data)

        data, slice = self.collate(graph_data_list)
        torch.save((data, slice), self.processed_paths[0])


class CyclicPepDatasetDropN(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None):
        super(CyclicPepDatasetDropN, self).__init__(root, transform, pre_transform)  # transform就是数据增强，对每一个数据都执行
        self.root = root
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['processed_cyc_data.dataset']

    def download(self):
        pass

    def process(self):
        graph_data_list = []

        # Assuming `data_cyc` is a list of tuples containing (id, pep_smile, Permeability)
        Permeability_values = [item[2] for item in tqdm(data_cyc, desc='Processing peptides')]
        normalized_values = normalize_data(Permeability_values)

        for idx, (id, pep_smile, permeability) in enumerate(tqdm(data_cyc, desc='Processing peptides')):
            molecule = Chem.MolFromSmiles(pep_smile)
            molecule = Chem.AddHs(molecule)
            AllChem.EmbedMolecule(molecule)
            data = mol_to_graph_data(molecule)
            data = drop_nodes(data)
            data.label = torch.tensor(normalized_values[idx], dtype=torch.float)
            data.label_true = torch.tensor(permeability, dtype=torch.float)

            graph_data_list.append(data)

        data, slice = self.collate(graph_data_list)
        torch.save((data, slice), self.processed_paths[0])


class CyclicPepDatasetDropNNoC(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None):
        super(CyclicPepDatasetDropNNoC, self).__init__(root, transform, pre_transform)  # transform就是数据增强，对每一个数据都执行
        self.root = root
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['processed_cyc_data.dataset']

    def download(self):
        pass

    def process(self):
        graph_data_list = []

        # Assuming `data_cyc` is a list of tuples containing (id, pep_smile, Permeability)
        Permeability_values = [item[2] for item in tqdm(data_cyc, desc='Processing peptides')]
        normalized_values = normalize_data(Permeability_values)

        for idx, (id, pep_smile, permeability) in enumerate(tqdm(data_cyc,
                                                                 desc='Processing peptides')):
            molecule = Chem.MolFromSmiles(pep_smile)
            molecule = Chem.AddHs(molecule)
            AllChem.EmbedMolecule(molecule)
            data = mol_to_graph_data(molecule)
            data = drop_nodes_nonC(data)
            data.label = torch.tensor(normalized_values[idx], dtype=torch.float)
            data.label_true = torch.tensor(permeability, dtype=torch.float)

            graph_data_list.append(data)

        data, slice = self.collate(graph_data_list)
        torch.save((data, slice), self.processed_paths[0])


class CyclicPepDatasetMaskN(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None):
        super(CyclicPepDatasetMaskN, self).__init__(root, transform, pre_transform)  # transform就是数据增强，对每一个数据都执行
        self.root = root
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['processed_cyc_data.dataset']

    def download(self):
        pass

    def process(self):
        graph_data_list = []

        # Assuming `data_cyc` is a list of tuples containing (id, pep_smile, Permeability)
        Permeability_values = [item[2] for item in tqdm(data_cyc, desc='Processing peptides')]
        normalized_values = normalize_data(Permeability_values)

        for idx, (id, pep_smile, permeability) in enumerate(tqdm(data_cyc, desc='Processing peptides')):
            molecule = Chem.MolFromSmiles(pep_smile)
            molecule = Chem.AddHs(molecule)
            AllChem.EmbedMolecule(molecule)
            data = mol_to_graph_data_maskn(molecule)
            data.label = torch.tensor(normalized_values[idx], dtype=torch.float)

            graph_data_list.append(data)

        data, slice = self.collate(graph_data_list)
        torch.save((data, slice), self.processed_paths[0])


class CyclicPepDatasetAddHydrogenBonds(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None):
        super(CyclicPepDatasetAddHydrogenBonds, self).__init__(root, transform, pre_transform)  # transform就是数据增强，对每一个数据都执行
        self.root = root
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['processed_cyc_data.dataset']

    def download(self):
        pass

    def process(self):
        graph_data_list = []

        # Assuming `data_cyc` is a list of tuples containing (id, pep_smile, Permeability)
        Permeability_values = [item[2] for item in tqdm(data_cyc, desc='Processing peptides')]
        # normalized_values = normalize_data(Permeability_values)

        for idx, (id, pep_smile, permeability) in enumerate(tqdm(data_cyc,
                                                                 desc='Processing peptides')):
            molecule = Chem.MolFromSmiles(pep_smile)
            molecule = Chem.AddHs(molecule)
            AllChem.EmbedMolecule(molecule)
            data = mol_to_graph_data_add__hydrogen_bonds(molecule)
            data.label = torch.tensor(Permeability_values[idx], dtype=torch.float)

            graph_data_list.append(data)

        data, slice = self.collate(graph_data_list)
        torch.save((data, slice), self.processed_paths[0])