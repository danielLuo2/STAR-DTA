"""Train STAR-DTA from scratch on the pancreatic-cancer ChEMBL set (ERBB2 / KRAS / TP53, drug-disjoint split).

Model selection and early stopping use the validation MSE; the test split is evaluated once at the end with the
selected checkpoint (linear output calibration fitted on training predictions).

    python train_pancreatic.py --data-dir data/pancreatic --seed 42
"""

import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from src.data import STARSampleDataset, load_pancreatic_split
from src.ema import ModelEMA
from src.model import STARDTA
from src.utils import set_seed
from train import evaluate, evaluate_calibrated, predict, standardize_samples


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=str, default="data/pancreatic")
    p.add_argument("--out-dir", type=str, default="runs/pancreatic")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--min-epochs", type=int, default=30)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.999)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    name = args.run_name or "star"
    out_dir = os.path.join(args.out_dir, f"{name}_seed{args.seed}_{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "args.txt"), "w", encoding="utf-8") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}={v}\n")

    train_s, val_s, test_s = load_pancreatic_split(args.data_dir)
    print(f"[INFO] pancreatic pairs train/val/test = {len(train_s)}/{len(val_s)}/{len(test_s)}")

    y_train = np.asarray([y for _, _, y in train_s], dtype=np.float32)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std() + 1e-8)
    train_s = standardize_samples(train_s, y_mean, y_std)
    val_s = standardize_samples(val_s, y_mean, y_std)
    test_s = standardize_samples(test_s, y_mean, y_std)

    train_ds, val_ds, test_ds = STARSampleDataset(train_s), STARSampleDataset(val_s), STARSampleDataset(test_s)
    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)
    train_eval_loader = DataLoader(train_ds, shuffle=False, **kw)

    model = STARDTA(hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    best_val, best_state, bad = float("inf"), None, 0
    best_path = os.path.join(out_dir, "best_model.pt")
    log_path = os.path.join(out_dir, "train_log.txt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(device)
            pred_aff, _, _ = model(batch)
            loss = F.mse_loss(pred_aff, batch.affinity.view(-1))
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            losses.append(float(loss.item()))
        scheduler.step()

        eval_model = ema.shadow if ema is not None else model
        val_metrics = evaluate(eval_model, val_loader, device, y_mean=y_mean, y_std=y_std)
        line = (
            f"epoch={epoch:03d} train_loss={np.mean(losses):.5f} "
            f"val_mse={val_metrics['mse']:.5f} val_ci={val_metrics['ci']:.5f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        if val_metrics["mse"] < best_val - 1e-6:
            best_val, bad = val_metrics["mse"], 0
            best_state = copy.deepcopy(eval_model.state_dict())
            torch.save({"model": best_state, "args": vars(args), "epoch": epoch, "y_mean": y_mean,
                        "y_std": y_std, "metrics": {"val": val_metrics}}, best_path)
            print(f"  [SAVE] val_mse={best_val:.5f}", flush=True)
        else:
            bad += 1
            if epoch >= args.min_epochs and args.patience > 0 and bad >= args.patience:
                print(f"[EARLY STOP] patience={args.patience} at epoch {epoch}")
                break

    # test once with the selected checkpoint
    eval_model.load_state_dict(best_state)
    test_metrics, y_test, p_test, _, _ = evaluate_calibrated(
        eval_model, train_eval_loader, test_loader, device, y_mean=y_mean, y_std=y_std
    )
    print("[TEST]", test_metrics)
    with open(os.path.join(out_dir, "best_metrics.txt"), "w", encoding="utf-8") as f:
        for k, v in test_metrics.items():
            f.write(f"{k}={v}\n")
    np.savez(os.path.join(out_dir, "best_predictions.npz"), y_test=y_test, p_test=p_test)
    print(f"[DONE] {out_dir}")


if __name__ == "__main__":
    main()
