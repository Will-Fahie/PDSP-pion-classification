#!/usr/bin/env python3
"""
Train one PFO classifier. Designed to be run several times in parallel, one
model per terminal.

    python utils/train_classifier.py --model hybrid --target photon --features all8
    python utils/train_classifier.py --model cnn    --target pion
    python utils/train_classifier.py --model mlp    --target photon --epochs 40

Why a script rather than the notebooks: six models need retraining on the
re-extracted data, and they are independent, so running them concurrently is
several times faster than working through the notebooks in sequence.

Outputs, per run:
    predictions/predictions-mc/<base>_<tag><suffix>.pkl   probs, labels, metrics
    models/<base>_<tag><suffix>.pt                        weights
    plots/<model>-classifier/<target>/history_<tag>.pkl   per-epoch history
    plots/<model>-classifier/<target>/timing_<tag>.json   wall-clock timing

`<tag>` is the feature set for the hybrid and the MLP, and 'seq' for the CNN,
which uses no summary features.

IMPORTANT when running in parallel: torch grabs every core it can see by
default, so N concurrent jobs each try to use all of them and spend their time
fighting. --threads caps each job; the default of 8 is sensible for 6 jobs on a
128-core machine.
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.general_utils import PROJECT_ROOT, target_config, purity, efficiency
from utils.evaluation_utils import save_results


# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------

BASE4 = ['track_chi2/ndof_proton', 'track_length', 'track_score', 'dEdX_median']

FEATURE_SETS = {
    'base4':      BASE4,
    'plus_bd':    BASE4 + ['b', 'd'],
    'plus_E':     BASE4 + ['shower_energy'],
    'plus_angle': BASE4 + ['beam_angle'],
    'all8':       BASE4 + ['b', 'd', 'shower_energy', 'beam_angle'],
}

MODEL_BASE = {'mlp': 'mlp_summary', 'cnn': 'cnn_dEdX', 'hybrid': 'hybrid_cnn_mlp'}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class PFODataset(Dataset):
    """One dataset for all three architectures. Each model takes
    (sequences, summary, mask) and ignores whichever it does not use, which is
    simpler than maintaining three dataset classes that differ only in what
    they drop."""

    def __init__(self, sequences, summary, masks, labels):
        self.sequences = torch.FloatTensor(sequences).permute(0, 2, 1)
        self.summary   = torch.FloatTensor(summary)
        self.masks     = torch.FloatTensor(masks)
        self.labels    = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.sequences[i], self.summary[i], self.masks[i], self.labels[i]


def load_data(cfg, feature_list, data_size="all"):
    prepared = PROJECT_ROOT / "prepared" / "prepared-mc"

    with open(prepared / f"train_{data_size}.pkl", "rb") as f:
        train = pickle.load(f)
    with open(prepared / "test.pkl", "rb") as f:
        test = pickle.load(f)

    # Column order comes from data-preparation via norm_stats, never hardcoded,
    # so this cannot drift from what was actually prepared.
    with open(prepared / "norm_stats.pkl", "rb") as f:
        all_features = pickle.load(f)["summary_features"]
    col = {k: i for i, k in enumerate(all_features)}

    missing = [f for f in feature_list if f not in col]
    if missing:
        raise SystemExit(f"features not in the prepared summary: {missing}\n"
                         f"available: {all_features}")
    cols = [col[k] for k in feature_list]

    return {
        'seq_train':  train["sequences"][:, :, 0:1],
        'seq_test':   test["sequences"][:, :, 0:1],
        'summ_train': train["summary"][:, cols],
        'summ_test':  test["summary"][:, cols],
        'mask_train': train["masks"],
        'mask_test':  test["masks"],
        'y_train':    train[cfg['label_key']],
        'y_test':     test[cfg['label_key']],
        'all_features': all_features,
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class MaskedGlobalAvgPool1d(nn.Module):
    def forward(self, x, mask):
        m = mask.unsqueeze(1)
        return (x * m).sum(dim=2) / m.sum(dim=2).clamp(min=1)


class FocalLoss(nn.Module):
    """Down-weights easy examples. With signal at 15-18% of the sample, plain
    BCE spends most of its gradient on background it already classifies."""

    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, targets, reduction='mean'):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        if reduction == 'none':
            return loss
        return loss.mean()


class MLPClassifier(nn.Module):
    """Summary statistics only. Ignores sequences and mask."""

    def __init__(self, n_summary=4, dropout=0.3, **_):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_summary, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 16),        nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, sequences, summary, mask):
        return self.net(summary).squeeze(1)


class PionCNN(nn.Module):
    """Hit-level dE/dx only. Ignores summary."""

    def __init__(self, in_channels=1, dropout=0.3, **_):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv1d(in_channels, 32, 3, padding=1), nn.BatchNorm1d(32)
        self.conv2, self.bn2 = nn.Conv1d(32, 64, 5, padding=2),          nn.BatchNorm1d(64)
        self.conv3, self.bn3 = nn.Conv1d(64, 64, 7, padding=3),          nn.BatchNorm1d(64)
        self.pool = MaskedGlobalAvgPool1d()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, sequences, summary, mask):
        x = sequences * mask.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(self.pool(x, mask))
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x).squeeze(1)


class HybridCNNMLP(nn.Module):
    """CNN branch over dE/dx fused with an MLP branch over the summary stats.
    n_summary comes from the data, so widening the summary array needs no
    change here."""

    def __init__(self, in_channels=1, n_summary=4, dropout=0.3, **_):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv1d(in_channels, 32, 3, padding=1), nn.BatchNorm1d(32)
        self.conv2, self.bn2 = nn.Conv1d(32, 64, 5, padding=2),          nn.BatchNorm1d(64)
        self.conv3, self.bn3 = nn.Conv1d(64, 64, 7, padding=3),          nn.BatchNorm1d(64)
        self.pool = MaskedGlobalAvgPool1d()

        self.mlp_branch = nn.Sequential(
            nn.Linear(n_summary, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 16),        nn.BatchNorm1d(16), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(80, 32), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, sequences, summary, mask):
        x = sequences * mask.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        cnn_feat = self.pool(x, mask)
        mlp_feat = self.mlp_branch(summary)
        return self.head(torch.cat([cnn_feat, mlp_feat], dim=1)).squeeze(1)


BUILDERS = {'mlp': MLPClassifier, 'cnn': PionCNN, 'hybrid': HybridCNNMLP}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def evaluate(model, loader, criterion, device, threshold=0.5):
    """Returns (mean_loss, loss_std, purity, efficiency, probs, ys).

    loss_std is the standard deviation of the PER-EXAMPLE loss across the
    whole epoch, not the batch-mean loss criterion normally returns -- that is
    what a loss error band needs. FocalLoss reduces internally (.mean()), so
    it is called with reduction='none' here and reduced by hand, using the
    same alpha/gamma the criterion was built with.
    """
    model.eval()
    all_losses = []
    probs, ys = [], []
    with torch.no_grad():
        for seq, summ, mask, y in loader:
            seq, summ, mask, y = (seq.to(device), summ.to(device),
                                  mask.to(device), y.to(device))
            logits = model(seq, summ, mask)
            per_example = criterion(logits, y, reduction='none')
            all_losses.append(per_example.cpu().numpy())
            probs.extend(torch.sigmoid(logits).cpu().numpy())
            ys.extend(y.cpu().numpy().astype(int))
    all_losses = np.concatenate(all_losses)
    probs, ys = np.array(probs), np.array(ys)
    preds = (probs >= threshold).astype(int)
    return (float(all_losses.mean()), float(all_losses.std(ddof=1)),
            purity(preds, ys, [1], [1]), efficiency(preds, ys, [1], [1]),
            probs, ys)


def train(model, loaders, device, n_epochs, patience, train_eval_every):
    """Returns (model, history, timing).

    Purity and efficiency are recorded for BOTH samples each epoch. The gap
    between them is the overfitting diagnostic -- the per-epoch analogue of
    Will's purity/efficiency against max depth.

    N is stored alongside so error bands can be drawn later without re-running.
    """
    train_loader, eval_train_loader, test_loader = loaders
    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5, patience=5)

    best_val, waited, best_state = float('inf'), 0, None
    history = {'train_loss': [], 'val_loss': [],
               'train_loss_std': [], 'val_loss_std': [],
               'train_loss_sem': [], 'val_loss_sem': [],
               'purity': [], 'efficiency': [],
               'train_purity': [], 'train_efficiency': [], 'train_epoch': [],
               'n_test_sig': 0, 'n_test_tagged': [],
               'n_train_sig': 0, 'n_train_tagged': []}

    n_test_total  = len(test_loader.dataset)
    n_train_total = len(train_loader.dataset)

    t0 = time.perf_counter()

    for epoch in range(n_epochs):
        model.train()
        batch_losses, batch_sizes = [], []
        for seq, summ, mask, y in train_loader:
            seq, summ, mask, y = (seq.to(device), summ.to(device),
                                  mask.to(device), y.to(device))
            optimizer.zero_grad()
            loss = criterion(model(seq, summ, mask), y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            batch_sizes.append(len(y))
        # tr is weighted by batch size, matching the original notebook's
        # total/n approach -- an unweighted mean of batch means overweights
        # the (usually smaller) last batch of an epoch.
        batch_losses = np.array(batch_losses)
        batch_sizes  = np.array(batch_sizes)
        tr = float((batch_losses * batch_sizes).sum() / batch_sizes.sum())
        # Std of per-BATCH means, not per-example -- the backward pass only
        # ever sees batch-mean loss, so that is what is available here without
        # a second forward pass. Coarser than the per-example std used for
        # val_loss_std below, but free. Unweighted here since it is only used
        # as a rough spread indicator, not as the reported mean.
        tr_std = float(batch_losses.std(ddof=1)) if len(batch_losses) > 1 else 0.0

        va, va_std, pur, eff, probs, ys = evaluate(model, test_loader, criterion, device)
        scheduler.step(va)

        history['train_loss'].append(tr)
        history['train_loss_std'].append(tr_std)
        history['train_loss_sem'].append(tr_std / np.sqrt(n_train_total))
        history['val_loss'].append(va)
        history['val_loss_std'].append(va_std)
        history['val_loss_sem'].append(va_std / np.sqrt(n_test_total))
        history['purity'].append(pur)
        history['efficiency'].append(eff)
        history['n_test_sig'] = int((ys == 1).sum())
        history['n_test_tagged'].append(int((probs >= 0.5).sum()))

        tr_pur = tr_eff = None
        # 0 = off entirely: skip the training-set pass, which is the ~2x cost.
        if train_eval_every > 0 and (epoch % train_eval_every == 0 or epoch == n_epochs - 1):
            _, _, tr_pur, tr_eff, tp, tys = evaluate(model, eval_train_loader,
                                                     criterion, device)
            history['train_purity'].append(tr_pur)
            history['train_efficiency'].append(tr_eff)
            history['train_epoch'].append(epoch)
            history['n_train_sig'] = int((tys == 1).sum())
            history['n_train_tagged'].append(int((tp >= 0.5).sum()))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            extra = (f" | train {100*tr_pur:.1f}% / {100*tr_eff:.1f}%"
                     if tr_pur is not None else "")
            print(f"  epoch {epoch+1:3d} | train {tr:.4f} | val {va:.4f} | "
                  f"purity {100*pur:.1f}% | efficiency {100*eff:.1f}%{extra}",
                  flush=True)

        if va < best_val:
            best_val, waited = va, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                print(f"  early stop at epoch {epoch+1}", flush=True)
                break

    elapsed = time.perf_counter() - t0
    done = len(history['train_loss'])
    model.load_state_dict(best_state)
    print(f"  best validation loss: {best_val:.4f}")
    print(f"  {done} epochs in {elapsed:.0f}s ({elapsed/done:.1f}s per epoch)")

    return model, history, {'seconds': elapsed, 'epochs': done,
                            's_per_epoch': elapsed / done,
                            'best_val_loss': best_val}


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True, choices=['mlp', 'cnn', 'hybrid'])
    p.add_argument('--target', required=True, choices=['pion', 'photon'])
    p.add_argument('--features', default='base4', choices=list(FEATURE_SETS),
                   help="Summary feature set. Ignored for --model cnn, which "
                        "uses no summary features. Default: base4.")
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--train-eval-every', type=int, default=0,
                   help="Evaluate the training set every N epochs, for the "
                        "overfitting check (train vs test purity/efficiency). "
                        "0 = off (default): training-set evaluation is skipped "
                        "entirely, which is what you want for routine "
                        "retraining. Evaluating every epoch roughly doubles "
                        "the time per epoch, since the training set is ~5x "
                        "the test set; use a larger N (e.g. 5) if you want the "
                        "check without paying that cost every epoch.")
    p.add_argument('--threads', type=int, default=8,
                   help="Cap torch threads. Without this, concurrent jobs each "
                        "grab every core and contend. Default: 8.")
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data-size', default='all')
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = target_config(args.target)
    base = MODEL_BASE[args.model]

    # The CNN uses no summary features, so labelling its output by a feature set
    # would be misleading.
    tag = 'seq' if args.model == 'cnn' else args.features
    feature_list = FEATURE_SETS[args.features]

    pred_dir  = PROJECT_ROOT / "predictions" / "predictions-mc"
    model_dir = PROJECT_ROOT / "models"
    plots_dir = PROJECT_ROOT / "plots" / f"{args.model}-classifier" / args.target
    for d in (pred_dir, model_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"{'='*70}")
    print(f"{args.model.upper()}  target={args.target}  tag={tag}")
    if args.model != 'cnn':
        print(f"features ({len(feature_list)}): {feature_list}")
    print(f"device={device}  threads={args.threads}  seed={args.seed}")
    print(f"{'='*70}", flush=True)

    d = load_data(cfg, feature_list, args.data_size)
    n_summary = d['summ_train'].shape[1]

    print(f"Train: {int(d['y_train'].sum()):,} signal / {len(d['y_train']):,} "
          f"({100*d['y_train'].mean():.1f}%)")
    print(f"Test:  {int(d['y_test'].sum()):,} signal / {len(d['y_test']):,} "
          f"({100*d['y_test'].mean():.1f}%)", flush=True)

    y_int = d['y_train'].astype(int)
    w = (1.0 / np.bincount(y_int))[y_int]
    sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)

    train_ds = PFODataset(d['seq_train'], d['summ_train'], d['mask_train'], d['y_train'])
    test_ds  = PFODataset(d['seq_test'],  d['summ_test'],  d['mask_test'],  d['y_test'])

    # train_loader uses the weighted sampler -- right for training, wrong for
    # measuring, since it resamples with replacement and rebalances the classes.
    # eval_train_loader passes over the same data sequentially and unweighted.
    loaders = (
        DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler),
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=False),
        DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False),
    )

    model = BUILDERS[args.model](n_summary=n_summary).to(device)
    print(f"parameters: {sum(q.numel() for q in model.parameters()):,}\n", flush=True)

    model, history, timing = train(model, loaders, device, args.epochs,
                                   args.patience, args.train_eval_every)

    _, _, _, _, probs, labels = evaluate(model, loaders[2], FocalLoss(gamma=2.0), device)

    # Threshold scan without plotting -- this runs headless.
    ts = np.arange(0.05, 0.96, 0.01)
    prod = [purity((probs >= t).astype(int), labels, [1], [1]) *
            efficiency((probs >= t).astype(int), labels, [1], [1]) for t in ts]
    best_t = float(ts[int(np.argmax(prod))])

    out_pred  = pred_dir  / f"{base}_{tag}{cfg['suffix']}.pkl"
    out_model = model_dir / f"{base}_{tag}{cfg['suffix']}.pt"

    res = save_results(probs, labels, best_t,
                       model_name=f"{args.model} {tag} -- {args.target} target",
                       save_path=out_pred)
    torch.save(model.state_dict(), out_model)

    with open(plots_dir / f"history_{args.model}_{tag}_{args.target}.pkl", "wb") as f:
        pickle.dump(history, f)
    with open(plots_dir / f"timing_{args.model}_{tag}_{args.target}.json", "w") as f:
        json.dump({**timing, 'model': args.model, 'target': args.target,
                   'tag': tag, 'features': feature_list if args.model != 'cnn' else [],
                   'n_summary': n_summary if args.model != 'cnn' else 0,
                   'device': str(device), 'threads': args.threads,
                   'n_train': int(len(d['y_train'])), 'n_test': int(len(d['y_test'])),
                   'batch_size': args.batch_size, 'seed': args.seed}, f, indent=2)

    print(f"\n{'='*70}")
    print(f"{args.model} {tag} {args.target}:  AUC {res['auc']:.3f}   "
          f"purity {100*res['purity']:.1f}%   "
          f"efficiency {100*res['efficiency']:.1f}%   t={best_t:.2f}")
    print(f"  {out_pred}")
    print(f"  {out_model}")
    print(f"  {plots_dir}/history_{args.model}_{tag}_{args.target}.pkl")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()