"""Evaluate a released STAR-DTA checkpoint on the test split it was trained for.

    python evaluate.py --checkpoint checkpoints/star-dta_davis_seed42.pt --data-root data
    python evaluate.py --checkpoint checkpoints/star-dta_pancreatic_seed42.pt --data-dir data/pancreatic

The checkpoint holds the evaluation (EMA) weights, the label standardization statistics and the linear output
calibration fitted on the training predictions, so the reported test metrics are reproduced directly.
"""

import argparse

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.data import STARSampleDataset, load_deepdta_split, load_pancreatic_split
from src.model import STARDTA
from train import metric_dict, predict


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default="data", help="Davis / KIBA directory (DeepDTA format)")
    p.add_argument("--data-dir", default="data/pancreatic", help="pancreatic-cancer set directory")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--save-predictions", default=None, help="optional .npz path for y_true / y_pred")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if cfg["dataset"] == "pancreatic":
        _, _, test = load_pancreatic_split(args.data_dir)
    else:
        _, test = load_deepdta_split(args.data_root, cfg["dataset"])
    y_mean, y_std = ckpt["y_mean"], ckpt["y_std"]
    test = [(s, t, (y - y_mean) / y_std) for s, t, y in test]
    loader = DataLoader(STARSampleDataset(test), batch_size=args.batch_size, shuffle=False)

    model = STARDTA(hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"]).to(device)
    model.load_state_dict(ckpt["model"])
    y, p_raw = predict(model, loader, device, y_mean=y_mean, y_std=y_std)
    p_cal = ckpt["cal_a"] * p_raw + ckpt["cal_b"]

    m = metric_dict(y, p_cal)
    print(f"{cfg['dataset']} seed={cfg['seed']} test: " + " ".join(f"{k}={v:.4f}" for k, v in m.items()))
    if "test_metrics" in ckpt:
        print("reported:", " ".join(f"{k}={ckpt['test_metrics'][k]:.4f}" for k in ["mse", "ci"]))
    if args.save_predictions:
        np.savez(args.save_predictions, y_true=y, y_pred=p_cal)


if __name__ == "__main__":
    main()
