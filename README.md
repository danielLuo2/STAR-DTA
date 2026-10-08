# STAR-DTA

**STAR-DTA: Scale-aware Tri-view Attention over Residues for Drug–Target Affinity Prediction**

STAR-DTA predicts drug–target binding affinity from a drug SMILES string and a protein sequence.

- **Drug encoder** — three views of the drug, fused into one vector:
  - a GIN over the molecular graph;
  - a character-level CNN over the SMILES string;
  - an MLP over the Morgan fingerprint.
- **Protein encoder** — a GraphDTA-style global convolution plus parallel residue convolutions with kernel sizes 5 / 7 / 9.
- **Interaction** — two cross-attention blocks:
  - the drug vector attends over residues;
  - a protein summary attends over the drug's atoms (SMILES tokens).

  The affinity head reads the two drug and protein vectors, their element-wise product and the two attention outputs.

## Repository layout

```text
train.py                 Davis / KIBA training and evaluation (standard split)
train_pancreatic.py      training on the pancreatic-cancer ChEMBL set
evaluate.py              evaluate a released checkpoint on its test split
src/model.py             the STAR-DTA model
src/data.py              data loading and the PyG sample dataset
src/features.py          molecular graph, SMILES, Morgan and protein featurization
src/utils.py, src/ema.py metrics (MSE, RMSE, CI, Pearson, Spearman), seeding, weight EMA
scripts/                 shell scripts for the paper's Table 1 (benchmark), Table 2 (pancreatic), Table 3 (ablation)
data/pancreatic/         the pancreatic-cancer ChEMBL set used in the paper
tools/                   script that built the pancreatic set from ChEMBL / UniProt
```

## Installation

The code was run with Python 3.10, PyTorch 2.11 (CUDA 12.8), PyTorch Geometric 2.7 and RDKit 2022.09 on one NVIDIA RTX 5090.

```bash
conda create -n star-dta python=3.10 -y
conda activate star-dta
# install PyTorch for your CUDA version first, see https://pytorch.org
pip install -r requirements.txt
```

## Data

**Davis and KIBA.**
- We use the standard split released with DeepDTA and also used by GraphDTA: `folds/train_fold_setting1.txt` and `folds/test_fold_setting1.txt`.
- Download the two dataset folders from the [DeepDTA repository](https://github.com/hkmztrk/DeepDTA/tree/master/data):

```bash
git clone --depth 1 https://github.com/hkmztrk/DeepDTA.git /tmp/DeepDTA
mkdir -p data && cp -r /tmp/DeepDTA/data/davis /tmp/DeepDTA/data/kiba data/
```

Each folder must contain `ligands_can.txt`, `proteins.txt`, `Y` and `folds/`. Davis Kd values are converted to pKd = −log10(Kd / 10⁹).

**Pancreatic-cancer set.**
- `data/pancreatic/` holds the 5,383 ChEMBL pairs (Ki / Kd / IC50 / EC50 as pChEMBL) for ERBB2, KRAS and TP53.
- The split is drug-disjoint: train / validation / test = 4,306 / 538 / 539.
- `tools/build_pancreatic_chembl.py` documents how the set was built. ChEMBL keeps growing, so please use the released files to reproduce the paper.
- The data come from [ChEMBL](https://www.ebi.ac.uk/chembl/) (CC BY-SA 3.0) and [UniProt](https://www.uniprot.org/) (CC BY 4.0).

## Reproducing the paper

All results use seeds 42, 2026 and 3407 and are reported as mean ± sample standard deviation.

```bash
bash scripts/run_benchmark.sh all     # Table 1: Davis and KIBA
bash scripts/run_pancreatic.sh        # Table 2: pancreatic-cancer set
bash scripts/run_ablation.sh          # Table 3: Davis ablations
```

A single run:

```bash
python train.py --dataset davis --data-root data --seed 42
python train_pancreatic.py --data-dir data/pancreatic --seed 42
```

**Training settings.** The defaults of `train.py` are the paper's settings:

| Setting | Value |
|---|---|
| Hidden size | 128 |
| Dropout | 0.15 |
| Optimizer | AdamW, learning rate 3e-4, weight decay 1e-5 |
| Schedule | 30-epoch linear warm-up, then cosine annealing |
| Batch size | 512 |
| Epochs | at most 1000 |
| Early stopping | patience 140, at least 180 epochs |
| Weight averaging | exponential moving average, decay 0.999 |
| Labels | standardized with the training mean and standard deviation |
| Output calibration | linear, y = a·p + b, fitted on training predictions |

**Outputs.** Each run writes `args.txt`, `train_log.txt`, `best_metrics.txt`, `best_model.pt` and `best_predictions.npz` to `runs/<dataset>/<name>_seed<seed>_<time>/`.

**Model selection.**
- The standard Davis / KIBA split has no validation fold. Following the GraphDTA protocol, the checkpoint with the lowest test-fold MSE is kept; the paper states this.
- On the pancreatic-cancer set, selection and early stopping use the validation split, and the test split is evaluated once at the end.

**Expected results (paper):**

| Dataset | MSE | CI |
|---|---|---|
| Davis | 0.217 ± 0.006 | 0.896 ± 0.005 |
| KIBA | 0.124 ± 0.002 | 0.904 ± 0.001 |
| Pancreatic-cancer set | 0.415 ± 0.010 | 0.810 ± 0.002 |

**Trained checkpoints.**
- The nine checkpoints behind the paper's numbers are attached to the [v1.0.0 release](https://github.com/danielLuo2/STAR-DTA/releases/tag/v1.0.0):
  - Davis, KIBA and the pancreatic-cancer set;
  - seeds 42, 2026 and 3407 each;
  - about 12 MB per file.
- Each file holds:
  - the evaluation (EMA) weights;
  - the label standardization statistics;
  - the output calibration.
- Put them in `checkpoints/` and run:

```bash
python evaluate.py --checkpoint checkpoints/star-dta_davis_seed42.pt --data-root data
python evaluate.py --checkpoint checkpoints/star-dta_pancreatic_seed42.pt --data-dir data/pancreatic
```

This prints the test metrics reported in the paper for that seed. `SHA256SUMS.txt` in the release lists the file checksums.

**Reproducibility.** PyTorch Geometric's GIN aggregation uses non-deterministic GPU kernels, so two runs with the same seed can differ slightly. Per-seed numbers will therefore not match exactly; the means and standard deviations should agree within the reported spread.

## Optional components

The model code also contains three extras. None of them is used in any result reported in the paper.

- **Drug-gated per-residue output** (`residue_head`).
- **Its residue-label loss** (`--region-labels`, `--lambda-region`). With no labels given, the loss weight is set to zero.
- **A seven-dimensional affinity-prior input branch** (`--affinity-prior`).

## Citation

The citation will be added after publication.

## License

Code: MIT License (see `LICENSE`). Data in `data/pancreatic/` keep the licenses of their sources (ChEMBL: CC BY-SA 3.0; UniProt: CC BY 4.0).
