"""Input featurization: molecular graph, SMILES tokens, Morgan fingerprint and protein tokens."""

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------- molecular graph (GraphDTA features)

ATOM_SYMBOLS = [
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe", "As", "Al", "I", "B",
    "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag", "Pd", "Co", "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu",
    "Au", "Ni", "Cd", "In", "Mn", "Zr", "Cr", "Pt", "Hg", "Pb", "Unknown",
]


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise ValueError(f"input {x} not in allowable set {allowable_set}")
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    """78-d atom features: element, degree, total H count, implicit valence, aromaticity."""
    return np.array(
        one_of_k_encoding_unk(atom.GetSymbol(), ATOM_SYMBOLS)
        + one_of_k_encoding(atom.GetDegree(), list(range(11)))
        + one_of_k_encoding_unk(atom.GetTotalNumHs(), list(range(11)))
        + one_of_k_encoding_unk(atom.GetImplicitValence(), list(range(11)))
        + [atom.GetIsAromatic()]
    )


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edges.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
    edge_index = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    return c_size, np.array(features, dtype=np.float32), edge_index


# ---------------------------------------------------------------- SMILES tokens

SMILES_VOCAB = (
    "0123456789"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "#=-()[]+@/\\:.%%"
)
SMILES_DICT = {c: i + 1 for i, c in enumerate(SMILES_VOCAB)}


def smiles_encode(smiles: str, max_len: int = 200) -> np.ndarray:
    """Character ids (0 = padding; characters outside the vocabulary map to one unknown id)."""
    x = np.zeros(max_len, dtype=np.int64)
    for i, ch in enumerate(smiles[:max_len]):
        x[i] = SMILES_DICT.get(ch, len(SMILES_DICT))
    return x


# ---------------------------------------------------------------- Morgan fingerprint

def morgan_fp_bits(smiles: str, n_bits: int = 1024, radius: int = 2) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


# ---------------------------------------------------------------- protein tokens (GraphDTA)

SEQ_VOC = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
SEQ_DICT = {v: (i + 1) for i, v in enumerate(SEQ_VOC)}


def seq_cat(prot, max_seq_len=1000):
    x = np.zeros(max_seq_len, dtype=np.int64)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = SEQ_DICT.get(ch, 0)
    return x
