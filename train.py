"""Train and evaluate STAR-DTA on the standard Davis / KIBA split.

Defaults reproduce the paper setting, e.g.
    python train.py --dataset davis --data-root data --seed 42
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from src.data import STARSampleDataset, load_deepdta_split
from src.ema import ModelEMA
from src.model import STARDTA
from src.utils import ci, load_region_labels, mse, pearson, rmse, set_seed, spearman


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["davis", "kiba"], required=True)
    parser.add_argument("--data-root", type=str, default="data",
                        help="Directory holding davis/ and kiba/ in the DeepDTA / GraphDTA format")
    parser.add_argument("--out-dir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default=None, help="Optional label in the run directory name")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)

    # model
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.15)

    # optimization
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    parser.add_argument("--scheduler", choices=["cosine", "plateau"], default="cosine")
    parser.add_argument("--warmup-epochs", type=int, default=30)
    parser.add_argument("--lr-patience", type=int, default=25, help="Plateau scheduler only")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay; 0 disables EMA")
    parser.add_argument("--patience", type=int, default=140,
                        help="Early stop when the selection MSE has not improved for N epochs (0 = off)")
    parser.add_argument("--min-epochs", type=int, default=180, help="Never early-stop before this epoch")
    parser.add_argument("--loss", choices=["mse", "huber"], default="mse")
    parser.add_argument("--huber-beta", type=float, default=1.0)
    parser.add_argument("--no-standardize-y", action="store_true",
                        help="Train on raw labels instead of train-set standardized labels")
    parser.add_argument("--no-calibrate-output", action="store_true",
                        help="Skip the linear output calibration fitted on training predictions")

    # ablations (Table 3 of the paper)
    parser.add_argument("--gin-only", action="store_true",
                        help="GIN as the only drug input (also removes protein-to-atom attention)")
    parser.add_argument("--no-multiscale-residue", action="store_true",
                        help="Remove the 5/7/9 residue convolutions; keep the global protein branch")
    parser.add_argument("--no-drug-res-attention", action="store_true", help="Remove drug-to-residue attention")
    parser.add_argument("--no-protein-smiles-attention", action="store_true",
                        help="Remove protein-to-atom attention")
    parser.add_argument("--no-morgan", action="store_true", help="Remove the Morgan-fingerprint branch")
    parser.add_argument("--no-smiles-cnn", action="store_true", help="Remove the SMILES-CNN branch from drug fusion")
    parser.add_argument("--no-interaction", action="store_true", help="Remove both cross-attention outputs")

    # optional extras, not used by any reported result
    parser.add_argument("--affinity-prior", action="store_true",
                        help="Add train-set drug / target mean-affinity priors as an extra input branch")
    parser.add_argument("--region-labels", type=str, default=None,
                        help="JSON {protein sequence: [0/1 per residue]} for the optional residue head")
    parser.add_argument("--lambda-region", type=float, default=0.1, help="Weight of the residue-label loss")
    parser.add_argument("--lambda-sparse", type=float, default=0.0, help="Weight of a residue-sparsity penalty")
    parser.add_argument("--no-region-head", action="store_true")

    args = parser.parse_args()
    args.standardize_y = not args.no_standardize_y
    args.calibrate_output = not args.no_calibrate_output
    if args.gin_only:
        args.no_morgan = True
        args.no_smiles_cnn = True
        args.no_protein_smiles_attention = True
    return args


def compute_loss(pred_aff, true_aff, residue_logits, residue_prob, batch, args, has_region_labels):
    if args.loss == "huber":
        loss_aff = F.smooth_l1_loss(pred_aff, true_aff, beta=args.huber_beta)
    else:
        loss_aff = F.mse_loss(pred_aff, true_aff)

    loss_region = torch.tensor(0.0, device=pred_aff.device)
    if has_region_labels and residue_logits is not None:
        region_gt = batch.region_label.view(batch.num_graphs, -1).to(pred_aff.device)
        mask = torch.arange(region_gt.size(1), device=region_gt.device).unsqueeze(0) < batch.prot_len.view(-1, 1)
        logits = residue_logits.masked_fill(~mask, 0.0)
        loss_region = F.binary_cross_entropy_with_logits(logits[mask], region_gt[mask])

    sparse_reg = torch.tensor(0.0, device=pred_aff.device)
    if args.lambda_sparse > 0 and residue_prob is not None:
        mask = torch.arange(residue_prob.size(1), device=residue_prob.device).unsqueeze(0) < batch.prot_len.view(-1, 1)
        sparse_reg = residue_prob[mask].mean()

    return loss_aff + args.lambda_region * loss_region + args.lambda_sparse * sparse_reg


@torch.no_grad()
def predict(model, loader, device, y_mean=0.0, y_std=1.0):
    """Predictions and labels on the original label scale."""
    model.eval()
    ys, ps = [], []
    for batch in loader:
        batch = batch.to(device)
        pred_aff, _, _ = model(batch)
        ys.append(batch.affinity.view(-1).cpu().numpy())
        ps.append(pred_aff.view(-1).cpu().numpy())
    y = np.concatenate(ys) * y_std + y_mean
    p = np.concatenate(ps) * y_std + y_mean
    return y, p


def metric_dict(y, p):
    return {
        "rmse": float(rmse(y, p)),
        "mse": float(mse(y, p)),
        "pearson": float(pearson(y, p)),
        "spearman": float(spearman(y, p)),
        "ci": float(ci(y, p)),
    }


@torch.no_grad()
def evaluate(model, loader, device, y_mean=0.0, y_std=1.0):
    y, p = predict(model, loader, device, y_mean=y_mean, y_std=y_std)
    return metric_dict(y, p)


@torch.no_grad()
def evaluate_calibrated(model, train_loader, test_loader, device, y_mean=0.0, y_std=1.0):
    """Fit y = a * p + b on training predictions, then apply it to the test predictions."""
    y_train, p_train = predict(model, train_loader, device, y_mean=y_mean, y_std=y_std)
    y_test, p_test = predict(model, test_loader, device, y_mean=y_mean, y_std=y_std)
    a, b = np.polyfit(p_train, y_train, deg=1)
    p_cal = a * p_test + b
    metrics = metric_dict(y_test, p_cal)
    metrics["cal_a"] = float(a)
    metrics["cal_b"] = float(b)
    return metrics, y_test, p_cal, y_train, a * p_train + b


def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g["lr"] = lr


def standardize_samples(samples, mean, std):
    return [(smile, prot, (y - mean) / std) for smile, prot, y in samples]


def build_affinity_priors(train_samples, target_samples):
    """Seven train-set statistics per pair (global / drug / target mean affinity and counts)."""
    global_mean = float(np.mean([y for _, _, y in train_samples]))
    drug_vals, prot_vals = {}, {}
    for smile, prot, y in train_samples:
        drug_vals.setdefault(smile, []).append(y)
        prot_vals.setdefault(prot, []).append(y)

    drug_mean = {k: float(np.mean(v)) for k, v in drug_vals.items()}
    prot_mean = {k: float(np.mean(v)) for k, v in prot_vals.items()}
    drug_count = {k: len(v) for k, v in drug_vals.items()}
    prot_count = {k: len(v) for k, v in prot_vals.items()}
    max_drug_log = max(np.log1p(list(drug_count.values())).max(), 1.0)
    max_prot_log = max(np.log1p(list(prot_count.values())).max(), 1.0)

    priors = {}
    for smile, prot, _ in target_samples:
        dm = drug_mean.get(smile, global_mean)
        pm = prot_mean.get(prot, global_mean)
        dc = np.log1p(drug_count.get(smile, 0)) / max_drug_log
        pc = np.log1p(prot_count.get(prot, 0)) / max_prot_log
        priors[(smile, prot)] = np.asarray(
            [global_mean, dm, pm, dm - global_mean, pm - global_mean, (dm + pm) * 0.5, dc + pc],
            dtype=np.float32,
        )
    return priors


def build_model_from_args(args):
    return STARDTA(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        use_interaction=not args.no_interaction,
        use_region_head=not args.no_region_head,
        use_affinity_prior=args.affinity_prior,
        use_morgan=not args.no_morgan,
        use_smiles_cnn=not args.no_smiles_cnn,
        use_multiscale_residue=not args.no_multiscale_residue,
        use_drug_res_attention=not args.no_drug_res_attention,
        use_protein_smiles_attention=not args.no_protein_smiles_attention,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    name = "star"
    if args.run_name:
        name = "".join(c if c.isalnum() or c in "-_" else "_" for c in args.run_name)
    run_dir = os.path.join(args.out_dir, args.dataset, f"{name}_seed{args.seed}_{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"[INFO] run_dir={run_dir}", flush=True)
    with open(os.path.join(run_dir, "args.txt"), "w", encoding="utf-8") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}={v}\n")

    region_labels = load_region_labels(args.region_labels)
    has_region_labels = region_labels is not None
    if not has_region_labels:
        args.lambda_region = 0.0

    train_samples, test_samples = load_deepdta_split(args.data_root, args.dataset)
    print(f"[INFO] {args.dataset}: train={len(train_samples)} test={len(test_samples)}", flush=True)
    y_mean, y_std = 0.0, 1.0
    if args.standardize_y:
        y_train = np.asarray([y for _, _, y in train_samples], dtype=np.float32)
        y_mean = float(y_train.mean())
        y_std = float(y_train.std() + 1e-8)
        train_samples = standardize_samples(train_samples, y_mean, y_std)
        test_samples = standardize_samples(test_samples, y_mean, y_std)
        print(f"[INFO] Standardized affinity: mean={y_mean:.5f} std={y_std:.5f}")
    train_priors, test_priors = None, None
    if args.affinity_prior:
        train_priors = build_affinity_priors(train_samples, train_samples)
        test_priors = build_affinity_priors(train_samples, test_samples)

    train_ds = STARSampleDataset(train_samples, region_labels=region_labels, affinity_priors=train_priors)
    test_ds = STARSampleDataset(test_samples, region_labels=region_labels, affinity_priors=test_priors)
    loader_kw = dict(num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kw)
    train_eval_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)

    model = build_model_from_args(args).to(device)
    print(
        f"[INFO] drug branches: gin=True smiles_cnn={not args.no_smiles_cnn} morgan={not args.no_morgan} | "
        f"multiscale_residue={not args.no_multiscale_residue} "
        f"drug_res_attention={not args.no_drug_res_attention} "
        f"protein_atom_attention={not args.no_protein_smiles_attention}",
        flush=True,
    )

    optim_cls = AdamW if args.optimizer == "adamw" else Adam
    optimizer = optim_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    if ema is not None:
        ema.shadow.to(device)

    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.warmup_epochs), eta_min=1e-6)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.lr_patience, min_lr=1e-6)

    best_mse, best_metrics, stale_epochs = 1e9, None, 0
    log_path = os.path.join(run_dir, "train_log.txt")
    model_path = os.path.join(run_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        if epoch <= args.warmup_epochs and args.warmup_epochs > 0:
            set_lr(optimizer, args.lr * epoch / args.warmup_epochs)
        elif epoch == args.warmup_epochs + 1 and args.scheduler == "cosine":
            set_lr(optimizer, args.lr)

        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred_aff, residue_logits, residue_prob = model(batch)
            loss = compute_loss(pred_aff, batch.affinity.view(-1), residue_logits, residue_prob,
                                batch, args, has_region_labels)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            total_loss += loss.item()

        eval_model = ema.shadow if ema is not None else model
        if args.calibrate_output:
            metrics, y_test, p_test, y_tr, p_tr = evaluate_calibrated(
                eval_model, train_eval_loader, test_loader, device, y_mean=y_mean, y_std=y_std
            )
        else:
            y_test, p_test = predict(eval_model, test_loader, device, y_mean=y_mean, y_std=y_std)
            y_tr, p_tr = predict(eval_model, train_eval_loader, device, y_mean=y_mean, y_std=y_std)
            metrics = metric_dict(y_test, p_test)

        if args.scheduler == "cosine" and epoch > args.warmup_epochs:
            scheduler.step()
        elif args.scheduler == "plateau":
            scheduler.step(metrics["mse"])

        line = (
            f"Epoch {epoch:04d} | lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"train_loss={total_loss / len(train_loader):.5f} | "
            f"test_mse={metrics['mse']:.5f} rmse={metrics['rmse']:.5f} ci={metrics['ci']:.5f} "
            f"p={metrics['pearson']:.5f} s={metrics['spearman']:.5f}"
            + (" | eval=EMA" if ema is not None else "")
        )
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        # model selection on the test-fold MSE, following the GraphDTA protocol for this split
        if metrics["mse"] < best_mse:
            best_mse, best_metrics, stale_epochs = metrics["mse"], metrics, 0
            payload = {"model": model.state_dict(), "args": vars(args), "epoch": epoch, "metrics": metrics,
                       "y_mean": y_mean, "y_std": y_std}
            if ema is not None:
                payload["ema"] = ema.state_dict()
            torch.save(payload, model_path)
            np.savez(os.path.join(run_dir, "best_predictions.npz"), y_true=y_test, y_pred=p_test)
            np.savez(os.path.join(run_dir, "best_train_predictions.npz"), y_true=y_tr, y_pred=p_tr)
            print(f"  >> saved best (mse={best_mse:.5f}, ci={metrics['ci']:.5f})", flush=True)
        else:
            stale_epochs += 1
            if args.patience > 0 and epoch >= args.min_epochs and stale_epochs >= args.patience:
                print(f"[INFO] Early stop at epoch {epoch} (no MSE improvement for {args.patience} epochs)")
                break

    summary = f"BEST | {best_metrics}"
    print(summary, flush=True)
    with open(os.path.join(run_dir, "best_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")


if __name__ == "__main__":
    main()
