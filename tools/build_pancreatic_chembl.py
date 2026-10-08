#!/usr/bin/env python3
"""Build the pancreatic-cancer ChEMBL set (ERBB2 / KRAS / TP53) used in the paper.

The files used in the paper are already in data/pancreatic/. ChEMBL keeps growing, so rerunning this
script today can return more pairs than the released set.

Aligns with DMFF-DTA case study (KEGG map05212 pathogenic targets).
Uses ChEMBL REST API (no chembl_webresource_client required).

Outputs under --out-dir (default: data/pancreatic/):
  sequences.json       UniProt sequences + ChEMBL target IDs
  chembl_pairs.csv     cleaned (smiles, target, pchembl, inchikey, ...)
  splits.json          drug-cold-start train/val/test InChIKey lists + indices
  fda_drugs.csv        (optional, --with-fda) approved small molecules with SMILES; not used in the paper

Example:
  python tools/build_pancreatic_chembl.py --out-dir data/pancreatic --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb"

# Affinity types aligned with DTA / DMFF-style ChEMBL cleaning
# Exclude POTENCY (inflates TP53 with non-classical affinity readouts)
KEEP_STANDARD_TYPES = {"KI", "KD", "IC50", "EC50"}

# Human UniProt accessions matching DMFF pancreatic case study
TARGETS = {
    "ERBB2": "P04626",
    "KRAS": "P01116",
    "TP53": "P04637",
}

# More tolerant SSL for flaky campus / server egress to EBI
_SSL_CTX = ssl.create_default_context()
try:
    _SSL_CTX.set_ciphers("DEFAULT:@SECLEVEL=1")
except ssl.SSLError:
    pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="data/pancreatic")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--page-size", type=int, default=500, help="smaller pages = fewer SSL mid-transfer fails")
    p.add_argument("--max-fda", type=int, default=3000, help="cap FDA molecules to download")
    p.add_argument("--with-fda", action="store_true", help="also download approved drugs (not used in the paper)")
    p.add_argument("--skip-chembl", action="store_true", help="reuse existing chembl_pairs.csv")
    p.add_argument(
        "--davis-proteins",
        type=str,
        default="data/davis/proteins.txt",
        help="GraphDTA-style proteins JSON for leakage/overlap note",
    )
    return p.parse_args()


def http_get_json(url: str, retries: int = 12, sleep: float = 3.0):
    """GET JSON with retries; handles transient SSL EOF from ChEMBL."""
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "STAR-DTA-pancreatic/1.0 (research; retry-friendly)",
                },
            )
            with urllib.request.urlopen(req, timeout=180, context=_SSL_CTX) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            last = e
            wait = sleep * (1.6 ** i) + (0.5 * i)
            wait = min(wait, 90.0)
            print(f"  [WARN] GET retry {i+1}/{retries} after {wait:.1f}s: {type(e).__name__}: {e}")
            time.sleep(wait)
    raise RuntimeError(f"GET failed after {retries} retries: {url}\n{last}")


def fetch_uniprot_sequence(accession: str) -> str:
    url = f"{UNIPROT_BASE}/{accession}.fasta"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=60) as resp:
        text = resp.read().decode("utf-8")
    lines = [ln.strip() for ln in text.splitlines() if ln and not ln.startswith(">")]
    seq = "".join(lines)
    if not seq:
        raise RuntimeError(f"empty UniProt sequence for {accession}")
    return seq


def resolve_human_target(accession: str) -> dict:
    """Find Homo sapiens SINGLE PROTEIN target whose component matches accession."""
    url = (
        f"{CHEMBL_BASE}/target.json?"
        + urllib.parse.urlencode(
            {
                "target_components__accession": accession,
                "organism": "Homo sapiens",
                "limit": 20,
            }
        )
    )
    data = http_get_json(url)
    targets = data.get("targets") or []
    # Prefer SINGLE PROTEIN
    singles = [t for t in targets if t.get("target_type") == "SINGLE PROTEIN"]
    pool = singles or targets
    if not pool:
        raise RuntimeError(f"No ChEMBL target for UniProt {accession}")
    # Prefer exact pref_name / shortest id heuristic: first SINGLE PROTEIN
    t = pool[0]
    return {
        "target_chembl_id": t["target_chembl_id"],
        "pref_name": t.get("pref_name"),
        "target_type": t.get("target_type"),
        "organism": t.get("organism"),
    }


def iter_activities(target_chembl_id: str, page_size: int = 1000):
    offset = 0
    while True:
        url = (
            f"{CHEMBL_BASE}/activity.json?"
            + urllib.parse.urlencode(
                {
                    "target_chembl_id": target_chembl_id,
                    "pchembl_value__isnull": "false",
                    "limit": page_size,
                    "offset": offset,
                }
            )
        )
        data = http_get_json(url)
        acts = data.get("activities") or []
        if not acts:
            break
        for a in acts:
            yield a
        meta = data.get("page_meta") or {}
        total = int(meta.get("total_count") or 0)
        offset += len(acts)
        print(f"  [activity] {target_chembl_id}: {offset}/{total}")
        if offset >= total or len(acts) < page_size:
            break
        time.sleep(0.2)


def canonicalize(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, None
    can = Chem.MolToSmiles(mol)
    ik = Chem.MolToInchiKey(mol)
    return can, ik


def davis_sequence_overlap(seq_meta: dict, davis_proteins_path):
    """Record exact sequence overlap with Davis proteins (leakage note for methods)."""
    report = {"davis_proteins_path": davis_proteins_path, "exact_matches": {}}
    if not davis_proteins_path or not os.path.isfile(davis_proteins_path):
        report["note"] = "Davis proteins file not found; skip overlap check"
        return report
    with open(davis_proteins_path, encoding="utf-8") as f:
        davis = json.load(f)
    # GraphDTA proteins.txt is often {name: sequence}
    davis_seqs = {str(v): str(k) for k, v in davis.items()}
    for gene, meta in seq_meta.items():
        name = davis_seqs.get(meta["sequence"])
        report["exact_matches"][gene] = name  # None if not in Davis
    n_hit = sum(1 for v in report["exact_matches"].values() if v)
    report["n_exact_hits"] = n_hit
    print(f"[INFO] Davis exact sequence overlap: {n_hit}/{len(seq_meta)} targets")
    for gene, name in report["exact_matches"].items():
        print(f"  {gene}: {'MATCH ' + name if name else 'no exact match'}")
    return report


def build_chembl_pairs(out_dir: str, page_size: int) -> pd.DataFrame:
    """Pull activities; SMILES come from activity.canonical_smiles (no per-mol fetch)."""
    seq_meta = {}
    rows = []
    for gene, acc in TARGETS.items():
        print(f"[INFO] resolving {gene} ({acc})")
        seq = fetch_uniprot_sequence(acc)
        tgt = resolve_human_target(acc)
        seq_meta[gene] = {
            "uniprot": acc,
            "sequence": seq,
            "length": len(seq),
            **tgt,
        }
        print(f"  -> {tgt['target_chembl_id']} ({tgt.get('pref_name')}), len={len(seq)}")

        n_raw = 0
        n_keep = 0
        for act in iter_activities(tgt["target_chembl_id"], page_size=page_size):
            n_raw += 1
            pchembl = act.get("pchembl_value")
            mid = act.get("molecule_chembl_id")
            smi = act.get("canonical_smiles")
            stype = (act.get("standard_type") or "").upper().strip()
            if pchembl is None or not mid or not smi:
                continue
            if stype not in KEEP_STANDARD_TYPES:
                continue
            # Prefer binding-like assays when annotated
            assay_type = (act.get("assay_type") or "").upper()
            if assay_type and assay_type not in {"B", "F", ""}:
                # keep B (binding) and F (functional); drop A/P/T noise when labeled
                if assay_type in {"A", "P", "T", "U"}:
                    continue
            try:
                pchembl = float(pchembl)
            except (TypeError, ValueError):
                continue
            can, ik = canonicalize(smi)
            if can is None:
                continue
            rows.append(
                {
                    "target_gene": gene,
                    "target_chembl_id": tgt["target_chembl_id"],
                    "molecule_chembl_id": mid,
                    "molecule_pref_name": act.get("molecule_pref_name"),
                    "pchembl": pchembl,
                    "standard_type": stype,
                    "assay_type": assay_type,
                    "assay_chembl_id": act.get("assay_chembl_id"),
                    "smiles": can,
                    "inchikey": ik,
                    "sequence": seq,
                }
            )
            n_keep += 1
        print(f"  [kept] {gene}: {n_keep}/{n_raw} activities (types={sorted(KEEP_STANDARD_TYPES)})")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No ChEMBL pairs collected")

    # Deduplicate: mean pChEMBL per (inchikey, target_gene)
    agg = (
        df.groupby(["inchikey", "target_gene"], as_index=False)
        .agg(
            smiles=("smiles", "first"),
            sequence=("sequence", "first"),
            pchembl=("pchembl", "mean"),
            molecule_chembl_id=("molecule_chembl_id", "first"),
            molecule_pref_name=("molecule_pref_name", "first"),
            target_chembl_id=("target_chembl_id", "first"),
            standard_type=("standard_type", "first"),
            n_assays=("pchembl", "count"),
        )
    )
    print(f"[INFO] pairs after dedup: {len(agg)} (raw={len(df)})")
    print(f"[INFO] per-target counts:\n{agg['target_gene'].value_counts().to_string()}")

    with open(os.path.join(out_dir, "sequences.json"), "w", encoding="utf-8") as f:
        json.dump(seq_meta, f, indent=2)
    agg.to_csv(os.path.join(out_dir, "chembl_pairs.csv"), index=False)
    return agg, seq_meta


def drug_cold_start_split(df: pd.DataFrame, seed: int, train_ratio: float, val_ratio: float):
    rng = np.random.default_rng(seed)
    drugs = df["inchikey"].unique()
    rng.shuffle(drugs)
    n = len(drugs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_d = set(drugs[:n_train])
    val_d = set(drugs[n_train : n_train + n_val])
    test_d = set(drugs[n_train + n_val :])

    def mask(s):
        return df["inchikey"].isin(s).to_numpy()

    splits = {
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "n_drugs": {"train": len(train_d), "val": len(val_d), "test": len(test_d)},
        "n_pairs": {},
        "train_inchikeys": sorted(train_d),
        "val_inchikeys": sorted(val_d),
        "test_inchikeys": sorted(test_d),
        "train_idx": df.index[mask(train_d)].tolist(),
        "val_idx": df.index[mask(val_d)].tolist(),
        "test_idx": df.index[mask(test_d)].tolist(),
    }
    for name, s in [("train", train_d), ("val", val_d), ("test", test_d)]:
        splits["n_pairs"][name] = int(mask(s).sum())
    print("[INFO] drug-cold-start split:", splits["n_drugs"], splits["n_pairs"])
    return splits


def fetch_fda_drugs(out_path: str, max_fda: int, page_size: int = 100):
    rows = []
    offset = 0
    while len(rows) < max_fda:
        url = (
            f"{CHEMBL_BASE}/molecule.json?"
            + urllib.parse.urlencode(
                {
                    "max_phase": 4,
                    "molecule_structures__isnull": "false",
                    "limit": page_size,
                    "offset": offset,
                }
            )
        )
        data = http_get_json(url)
        mols = data.get("molecules") or []
        if not mols:
            break
        for mol in mols:
            structs = mol.get("molecule_structures") or {}
            smi = structs.get("canonical_smiles")
            if not smi:
                continue
            # Prefer small molecules
            mol_type = (mol.get("molecule_type") or "").lower()
            if mol_type and mol_type not in {"small molecule", ""}:
                continue
            can, ik = canonicalize(smi)
            if can is None:
                continue
            # Soft drug-likeness filter (optional but keeps docking/physchem sane)
            m = Chem.MolFromSmiles(can)
            mw = Descriptors.MolWt(m)
            if mw < 100 or mw > 900:
                continue
            rows.append(
                {
                    "molecule_chembl_id": mol.get("molecule_chembl_id"),
                    "pref_name": mol.get("pref_name"),
                    "smiles": can,
                    "inchikey": ik,
                    "max_phase": mol.get("max_phase"),
                    "molecular_weight": mw,
                }
            )
            if len(rows) >= max_fda:
                break
        meta = data.get("page_meta") or {}
        total = int(meta.get("total_count") or 0)
        offset += len(mols)
        print(f"  [FDA] collected={len(rows)} scanned={offset}/{total}")
        if offset >= total or len(mols) < page_size:
            break
        time.sleep(0.2)

    df = pd.DataFrame(rows).drop_duplicates(subset=["inchikey"])
    df.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path} ({len(df)} FDA drugs)")
    return df


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pairs_path = os.path.join(args.out_dir, "chembl_pairs.csv")
    seq_path = os.path.join(args.out_dir, "sequences.json")
    if args.skip_chembl and os.path.isfile(pairs_path) and os.path.isfile(seq_path):
        df = pd.read_csv(pairs_path)
        with open(seq_path, encoding="utf-8") as f:
            seq_meta = json.load(f)
        print(f"[OK] reuse {pairs_path} ({len(df)})")
    else:
        df, seq_meta = build_chembl_pairs(args.out_dir, args.page_size)

    # Reset index for stable split indices
    df = df.reset_index(drop=True)
    df.to_csv(pairs_path, index=False)

    overlap = davis_sequence_overlap(seq_meta, args.davis_proteins)

    splits = drug_cold_start_split(df, args.seed, args.train_ratio, args.val_ratio)
    splits_path = os.path.join(args.out_dir, "splits.json")
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    print(f"[OK] wrote {splits_path}")

    if args.with_fda:
        fetch_fda_drugs(os.path.join(args.out_dir, "fda_drugs.csv"), args.max_fda)

    meta = {
        "targets": TARGETS,
        "n_pairs": len(df),
        "n_drugs": int(df["inchikey"].nunique()),
        "per_target": {g: int((df["target_gene"] == g).sum()) for g in TARGETS},
        "split": splits["n_pairs"],
        "label": "pChEMBL (higher = stronger binder)",
        "keep_standard_types": sorted(KEEP_STANDARD_TYPES),
        "davis_sequence_overlap": overlap,
        "note": "ChEMBL Ki/Kd/IC50/EC50 only (POTENCY excluded). Larger than DMFF ~2889 due to ChEMBL growth; TP53 remains sparse.",
    }
    with open(os.path.join(args.out_dir, "dataset_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("[DONE] pancreatic ChEMBL dataset ready:", args.out_dir)


if __name__ == "__main__":
    main()
