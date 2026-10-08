"""Dataset loading (Davis / KIBA standard split, pancreatic ChEMBL set) and the PyG sample dataset."""

import json
import os
import pickle
from collections import OrderedDict

import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.data import Data

from src.features import morgan_fp_bits, seq_cat, smile_to_graph, smiles_encode


def load_deepdta_split(data_root, dataset_name):
    """Standard Davis / KIBA split released with DeepDTA and used by GraphDTA (folds/*_fold_setting1.txt).

    Returns train and test lists of (canonical SMILES, protein sequence, affinity); Davis Kd values are
    converted to pKd = -log10(Kd / 1e9).
    """
    fpath = os.path.join(data_root, dataset_name)
    with open(os.path.join(fpath, "folds", "train_fold_setting1.txt")) as f:
        train_fold = [ee for e in json.load(f) for ee in e]
    with open(os.path.join(fpath, "folds", "test_fold_setting1.txt")) as f:
        test_fold = json.load(f)
    with open(os.path.join(fpath, "ligands_can.txt")) as f:
        ligands = json.load(f, object_pairs_hook=OrderedDict)
    with open(os.path.join(fpath, "proteins.txt")) as f:
        proteins = json.load(f, object_pairs_hook=OrderedDict)
    with open(os.path.join(fpath, "Y"), "rb") as f:
        affinity = pickle.load(f, encoding="latin1")

    drugs = [Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True) for d in ligands.keys()]
    prots = [proteins[t] for t in proteins.keys()]

    affinity = np.asarray(affinity)
    if dataset_name == "davis":
        affinity = -np.log10(affinity / 1e9)

    rows, cols = np.where(np.isnan(affinity) == False)  # noqa: E712
    train_rows, train_cols = rows[train_fold], cols[train_fold]
    test_rows, test_cols = rows[test_fold], cols[test_fold]

    train_samples = [(drugs[r], prots[c], float(affinity[r, c])) for r, c in zip(train_rows, train_cols)]
    test_samples = [(drugs[r], prots[c], float(affinity[r, c])) for r, c in zip(test_rows, test_cols)]
    return train_samples, test_samples


def load_pancreatic_split(data_dir):
    """Pancreatic-cancer ChEMBL set (ERBB2 / KRAS / TP53), drug-disjoint train / val / test split.

    Returns train, val, test lists of (SMILES, protein sequence, pChEMBL).
    """
    import pandas as pd

    pairs = pd.read_csv(os.path.join(data_dir, "chembl_pairs.csv"))
    with open(os.path.join(data_dir, "splits.json"), encoding="utf-8") as f:
        splits = json.load(f)

    def to_samples(idxs):
        rows = pairs.iloc[idxs]
        return [(str(r.smiles), str(r.sequence), float(r.pchembl)) for r in rows.itertuples(index=False)]

    return to_samples(splits["train_idx"]), to_samples(splits["val_idx"]), to_samples(splits["test_idx"])


class STARSampleDataset(Dataset):
    """One PyG Data object per drug-target pair (graph, SMILES ids, Morgan bits, protein tokens, label)."""

    def __init__(self, samples, region_labels=None, max_seq_len=1000, affinity_priors=None):
        self.samples = samples
        self.region_labels = region_labels
        self.max_seq_len = max_seq_len
        self.affinity_priors = affinity_priors or {}
        self.smile_graph_cache = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        smile, prot_seq, y = self.samples[idx]
        if smile not in self.smile_graph_cache:
            self.smile_graph_cache[smile] = smile_to_graph(smile)
        _, x, edge_index = self.smile_graph_cache[smile]

        data = Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
        )
        # [1, L] so that PyG batching stacks to [B, L]
        data.protein = torch.tensor(seq_cat(prot_seq, self.max_seq_len), dtype=torch.long).unsqueeze(0)
        data.affinity = torch.tensor([y], dtype=torch.float32)
        data.prot_len = torch.tensor([min(len(prot_seq), self.max_seq_len)], dtype=torch.long)
        data.smiles_ids = torch.tensor(smiles_encode(smile), dtype=torch.long).unsqueeze(0)
        data.morgan_fp = torch.tensor(morgan_fp_bits(smile), dtype=torch.float32).unsqueeze(0)
        prior = self.affinity_priors.get((smile, prot_seq), np.zeros(7, dtype=np.float32))
        data.affinity_prior = torch.tensor(prior, dtype=torch.float32).unsqueeze(0)
        data.region_label = None

        if self.region_labels is not None and prot_seq in self.region_labels:
            label = np.array(self.region_labels[prot_seq], dtype=np.float32)[: self.max_seq_len]
            padded = np.zeros(self.max_seq_len, dtype=np.float32)
            padded[: len(label)] = label
            data.region_label = torch.tensor(padded, dtype=torch.float32).unsqueeze(0)
        return data
