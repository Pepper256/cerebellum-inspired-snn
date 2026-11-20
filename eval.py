"""Evaluate a trained cerebellum-inspired SNN checkpoint on MNIST.

Inference only: one forward pass, no weight changes. Safe to run any time.

    python eval.py                      # runs/main/best.pt
    python eval.py --ckpt runs/main/best.pt --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data import get_loaders
from model import CerebellarSNN

RUNS = Path(__file__).resolve().parent / "runs"


def build(a: dict) -> CerebellarSNN:
    """Rebuild from the checkpoint's stored args.

    Uses .get() with the current defaults so that checkpoints written before an
    argument was renamed still load (e.g. the GoC knobs became goc_gain/goc_tau).
    """
    d = dict(n_grc=4096, grc_fan_in=8, n_goc_groups=16, n_group=1,
             n_pc_per_class=1, T=25, beta=0.9, thresh=1.0, reset="subtract",
             encoder="direct", grc_gain=1.0, pc_gain=5.0, goc_gain=2.0,
             goc_tau=2.0, gamma=3.0, readout="quadratic", mfd_dcn="tonic",
             pc_homeo=True, seed=0,
             # v4 (MLI). Absent from old checkpoints -> defaults to 0 = v3, so
             # every frozen baseline still rebuilds bit-identically.
             n_mli=0, mli_fan_in=16, mli_pc_fan_in=8, mli_in_gain=4.0,
             mli_gain=3.0, mli_beta=0.9, mli_thresh=1.0)
    d.update({k: v for k, v in a.items() if k in d})
    return CerebellarSNN(**d)


@torch.no_grad()
def run(model, loader, device) -> dict:
    correct = total = 0
    grc = 0.0
    mli = 0.0
    pc = [0.0] * model.n_group
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, st = model(x, return_stats=True)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
        grc += st["grc_rate"] * y.numel()
        if model.n_mli:
            mli += st["mli_rate"] * y.numel()
        for e in range(model.n_group):
            pc[e] += st["pc_rate_per_group"][e] * y.numel()
    return {"acc": correct / max(total, 1), "n": total,
            "grc_rate": grc / max(total, 1),
            "mli_rate": mli / max(total, 1),
            "pc_rates": [v / max(total, 1) for v in pc]}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(RUNS / "main" / "best.pt"))
    p.add_argument("--split", default="test", choices=["val", "test", "train"])
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default=None)
    a = p.parse_args()

    dev = torch.device(a.device) if a.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    ckpt = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model = build(ckpt["args"]).to(dev)
    sd = dict(ckpt["model"])
    # ---- backward compatibility ------------------------------------------------
    # v1 checkpoints predate `goc_g`; v2 stored it as ONE scalar, but v3 uses one
    # gain per GoC group.  Expand the scalar so old checkpoints still load (all
    # groups equal -> identical behaviour).
    if "goc_g" in sd and sd["goc_g"].dim() == 0:
        sd["goc_g"] = sd["goc_g"].expand(model.n_goc_groups).clone()
    allowed_missing = {"goc_g", "grc_rate_ema"}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    extra = [k for k in missing if k not in allowed_missing]
    if extra or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing} "
                           f"unexpected={unexpected}")
    model.eval()

    loaders = get_loaders(batch_size=a.batch_size, val_size=5000,
                          seed=ckpt["args"]["seed"])
    res = run(model, loaders[a.split], dev)
    print(f"checkpoint : {a.ckpt}  (epoch {ckpt['epoch']}, "
          f"val {ckpt['val_acc'] * 100:.2f}%)")
    print(f"split      : {a.split}  n={res['n']}")
    print(f"accuracy   : {res['acc'] * 100:.2f}%")
    print(f"GrC rate   : {res['grc_rate']:.4f}")
    if model.n_mli:
        print(f"MLI rate   : {res['mli_rate']:.4f}")
    print(f"PC rates   : " + "  ".join(f"g{i}={v:.4f}" for i, v in enumerate(res["pc_rates"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
