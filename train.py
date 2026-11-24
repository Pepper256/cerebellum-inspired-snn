"""Train the cerebellum-inspired SNN on MNIST with local rules (no backprop).

!! The agent must NOT run this file without explicit user authorisation. !!
`--dry-run` (random tensors, no dataset, no weight change beyond one probe update)
is safe and is the recommended smoke test.

Protocol is deliberately identical to 01/02/03 so the numbers are comparable:
    * MNIST only, 55k train / 5k val / 10k test
    * T = 25 simulation steps, direct (constant-current) encoding
    * best checkpoint selected on the 5k val split; test is never used for selection
    * reported number is test(10k) accuracy + spike statistics for the energy view

Examples
--------
    python train.py --dry-run                          # 6-point self check, no data
    python train.py --epochs 30                        # MAIN run
    python train.py --epochs 30 --no-cf                # ablation a: no teaching signal
    python train.py --epochs 30 --groups g0            # ablation b: LTD only
    python train.py --epochs 30 --goc-target 1.0       # ablation c: GoC off
    python train.py --epochs 30 --mfd-dcn none         # ablation d: no MF->DCN bias
    python train.py --epochs 30
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from data import get_loaders
from model import CerebellarSNN
from rules import CaveConfig, cave_update, grc_update, goc_homeo_update

RUNS = Path(__file__).resolve().parent / "runs"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="cerebellum-inspired SNN on MNIST")
    # ---- architecture ----
    p.add_argument("--n-grc", type=int, default=4096)
    p.add_argument("--grc-fan-in", type=int, default=8)
    p.add_argument("--n-goc-groups", type=int, default=16)
    p.add_argument("--n-group", type=int, default=1)
    p.add_argument("--n-pc-per-class", type=int, default=1)
    p.add_argument("--T", type=int, default=25)
    p.add_argument("--beta", type=float, default=0.9)
    p.add_argument("--thresh", type=float, default=1.0)
    p.add_argument("--reset", default="subtract", choices=["subtract", "zero"])
    p.add_argument("--encoder", default="direct", choices=["direct", "poisson"])
    p.add_argument("--grc-gain", type=float, default=1.0)
    p.add_argument("--pc-gain", type=float, default=5.0)
    p.add_argument("--goc-gain", type=float, default=2.0,
                   help="strength of GoC feedback inhibition (sparsity knob)")
    p.add_argument("--goc-tau", type=float, default=2.0,
                   help="low-pass time constant of the GoC rate estimate")
    p.add_argument("--gamma", type=float, default=3.0)
    p.add_argument("--readout", default="quadratic", choices=["quadratic", "linear"])
    p.add_argument("--mfd-dcn", default="tonic", choices=["tonic", "none", "full"])
    p.add_argument("--no-pc-homeo", action="store_true",
                   help="ablation: disable intrinsic plasticity on the PC gains")
    p.add_argument("--grc-learn", default="off", choices=["off", "hebb"],
                   help="unsupervised plasticity at MF->GrC (hebb = competitive "
                        "learning with synaptic normalisation; no labels, no CF)")
    p.add_argument("--eta-g", type=float, default=0.02, help="GrC learning rate")
    p.add_argument("--goc-homeo", action="store_true",
                   help="adapt the GoC gain to hold granule activity at --goc-target")
    p.add_argument("--eta-goc", type=float, default=0.5)
    p.add_argument("--goc-target", type=float, default=0.05)
    p.add_argument("--goc-gate", action="store_true",
                   help="v3: let GoC state gate the DIRECTION of MF->GrC plasticity "
                        "(BCM-style: over-active cell -> LTD, quiet cell -> LTP)")
    p.add_argument("--gate-kappa", type=float, default=1.0)
    p.add_argument("--gate-scale", type=float, default=0.02)
    # ---- learning rules ----
    p.add_argument("--eta0", type=float, default=0.3)
    p.add_argument("--eta1", type=float, default=0.3)
    p.add_argument("--eta-b", type=float, default=0.01)
    p.add_argument("--eta-v", type=float, default=0.01)
    p.add_argument("--group1-scope", default="argmax", choices=["argmax", "all"])
    p.add_argument("--eta-h", type=float, default=0.5, help="intrinsic-plasticity rate")
    p.add_argument("--rate-target", type=float, default=0.10, help="PC rate set point")
    p.add_argument("--no-norm", action="store_true", help="ablation g: no homeostatic renorm")
    p.add_argument("--no-cf", action="store_true", help="ablation a: no teaching signal")
    p.add_argument("--groups", default="bidir",
                   choices=["bidir", "bidir_all", "both", "g0", "g1"],
                   help="bidir: one matrix gets both directions (has a fixed point); "
                        "bidir_all: same rule on every microzone (ensemble); "
                        "both/g0/g1: the two-separate-matrices variants (ruled out)")
    # ---- optimisation / io ----
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    return p


def make_model(a: argparse.Namespace) -> CerebellarSNN:
    return CerebellarSNN(
        n_grc=a.n_grc, grc_fan_in=a.grc_fan_in, n_goc_groups=a.n_goc_groups,
        n_group=a.n_group, n_pc_per_class=a.n_pc_per_class,
        T=a.T, beta=a.beta, thresh=a.thresh, reset=a.reset, encoder=a.encoder,
        grc_gain=a.grc_gain, pc_gain=a.pc_gain,
        goc_gain=a.goc_gain, goc_tau=a.goc_tau,
        gamma=a.gamma, readout=a.readout, mfd_dcn=a.mfd_dcn,
        pc_homeo=not a.no_pc_homeo, grc_learn=a.grc_learn, seed=a.seed,
    )


def make_cfg(a: argparse.Namespace) -> CaveConfig:
    return CaveConfig(eta0=a.eta0, eta1=a.eta1, eta_b=a.eta_b, eta_v=a.eta_v,
                      group1_scope=a.group1_scope, normalise=not a.no_norm,
                      use_cf=not a.no_cf, groups=a.groups,
                      pc_homeo=not a.no_pc_homeo, eta_h=a.eta_h,
                      rate_target=a.rate_target, grc_learn=a.grc_learn,
                      eta_g=a.eta_g, goc_homeo=a.goc_homeo,
                      eta_goc=a.eta_goc, goc_target=a.goc_target,
                      goc_gate=a.goc_gate, gate_kappa=a.gate_kappa,
                      gate_scale=a.gate_scale)


def pick_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    correct = total = 0
    grc = 0.0
    pc = [0.0] * model.n_group
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, st = model(x, return_stats=True)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
        grc += st["grc_rate"] * y.numel()
        for e in range(model.n_group):
            pc[e] += st["pc_rate_per_group"][e] * y.numel()
    return {"acc": correct / max(total, 1),
            "grc_rate": grc / max(total, 1),
            "pc_rates": [v / max(total, 1) for v in pc]}


def main() -> int:
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    model = make_model(args).to(device)
    cfg = make_cfg(args)
    print(f"device={device}  synapses={model.n_synapses():,}  "
          f"trainable={model.n_trainable():,}  T={args.T}")
    print(f"  {model.extra_repr()}")
    print(f"  rules: eta0={cfg.eta0} eta1={cfg.eta1} scope={cfg.group1_scope} "
          f"norm={cfg.normalise} cf={cfg.use_cf} groups={cfg.groups}")

    if args.dry_run:
        # real MNIST, not torch.rand: uniform noise has mean 0.5 vs MNIST 0.13 and
        # saturates the population, which makes every activity statistic useless.
        # NOTE: no local `import` here -- that would shadow the module-level
        # get_loaders and make the training branch below raise UnboundLocalError.
        loaders = get_loaders(batch_size=64, val_size=0, num_workers=0)
        x, y = next(iter(loaders["test"]))
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits, st = model(x, return_stats=True)
        cnt = st["grc_rate_per_neuron"]
        cv = (cnt.std() / cnt.mean().clamp_min(1e-9)).item()
        print("\ndry-run forward ok on a real MNIST batch (no weight update)")
        print(f"  GrC rate {st['grc_rate']:.4f}  CV {cv:.3f}  "
              f"silent {(cnt == 0).float().mean() * 100:.1f}%")
        print(f"  PC rates per group: "
              + "  ".join(f"{v:.4f}" for v in st["pc_rate_per_group"]))
        print(f"  pooled I per group: "
              + "  ".join(f"{v:.4f}" for v in st["I_mean_per_group"]))
        print(f"  GoC inhibition mean {st['goc_inh_mean']:.4f} max {st['goc_inh_max']:.4f}")
        print(f"  logits std {logits.std():.4f} range "
              f"[{logits.min():+.3f}, {logits.max():+.3f}]")
        with torch.no_grad():
            diag = cave_update(model, st["grc_counts"] / args.T, logits, y, cfg, x=x,
                               pc_rate=st["pc_rate_per_neuron"])
        print(f"  one probe update: wrong {diag['n_wrong']}/{y.numel()}  "
              f"|dW0| {diag['dW0_abs']:.2e}  |dW1| {diag['dW1_abs']:.2e}")
        return 0

    out_dir = Path(args.out_dir) if args.out_dir else RUNS / "main"
    out_dir.mkdir(parents=True, exist_ok=True)
    loaders = get_loaders(batch_size=args.batch_size, val_size=args.val_size,
                          seed=args.seed)

    csv_path = out_dir / "metrics.csv"
    best_val, best_epoch, best_state = -1.0, -1, None
    t0 = time.time()
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_acc", "grc_rate",
                         "pc_rate_g0", "pc_rate_g1", "n_wrong", "elapsed_s"])
        for epoch in range(1, args.epochs + 1):
            model.train()
            correct = total = wrong = 0
            gdiag_acc = {"grc_dW": 0.0, "grc_row_max_mean": float("nan"),
                         "grc_rows_updated": 0, "grc_gate_mean": 0.0}
            grc_sum = 0.0
            for i, (x, y) in enumerate(loaders["train"]):
                if args.max_batches and i >= args.max_batches:
                    break
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    logits, st = model(x, return_stats=True)
                    z = st["grc_counts"] / args.T
                    # UNSUPERVISED granule-layer plasticity first (no labels, no CF),
                    # then the CF-gated Purkinje update
                    gdiag = grc_update(model, x, st["grc_counts"], cfg)
                    gdiag.update(goc_homeo_update(model, st["grc_rate_per_group"], cfg))
                    diag = cave_update(model, z, logits, y, cfg, x=x,
                                       pc_rate=st["pc_rate_per_neuron"])
                correct += (logits.argmax(1) == y).sum().item()
                total += y.numel()
                wrong += diag["n_wrong"]
                gdiag_acc = gdiag
                grc_sum += st["grc_rate"] * y.numel()

            tr_acc = correct / max(total, 1)
            val = evaluate(model, loaders["val"], device) if loaders["val"] else {"acc": float("nan")}
            val_acc = val["acc"]
            elapsed = time.time() - t0
            writer.writerow([epoch, "", f"{tr_acc:.4f}", f"{val_acc:.4f}",
                             f"{grc_sum / max(total, 1):.4f}",
                             f"{val.get('pc_rates', [float('nan')])[0]:.4f}",
                             f"{val.get('pc_rates', [float('nan'), float('nan')])[1]:.4f}"
                             if model.n_group > 1 else "",
                             wrong, f"{elapsed:.1f}"])
            fh.flush()
            print(f"epoch {epoch:2d}/{args.epochs}  train {tr_acc * 100:.2f}%  "
                  f"val {val_acc * 100:.2f}%  grc {grc_sum / max(total, 1):.3f}  "
                  + f"wrong {wrong}/{total}"
                  + (f"  grc_dW {gdiag_acc['grc_dW']:.1e}  rowmax "
                     f"{gdiag_acc['grc_row_max_mean']:.3f}  upd "
                     f"{gdiag_acc['grc_rows_updated']}"
                     + (f"  gate {gdiag_acc.get('grc_gate_mean', 0):+.2f}"
                        if args.goc_gate else "")
                     if args.grc_learn != "off" else "")
                  + f"  [{elapsed:.0f}s]", flush=True)

            if val_acc == val_acc and val_acc > best_val:
                best_val, best_epoch = val_acc, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                torch.save({"model": best_state, "args": vars(args), "epoch": epoch,
                            "val_acc": val_acc}, out_dir / "best.pt")

    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate(model, loaders["test"], device)
    summary = {
        "arch": "cerebellum",
        "best_epoch": best_epoch,
        "best_val_acc": best_val,
        "test_acc": test["acc"],
        "grc_rate": test["grc_rate"],
        "pc_rates": test["pc_rates"],
        "synapses": float(model.n_synapses()),
        "trainable": float(model.n_trainable()),
        "params": float(model.n_params()),
        "wall_clock_s": round(time.time() - t0, 1),
        "protocol": "local rules, no autograd; best on 5k val split",
        "config": {k: v for k, v in vars(args).items() if k != "out_dir"},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== test ===")
    print(f"accuracy        : {test['acc'] * 100:.2f}%  "
          f"(best val {best_val * 100:.2f}% @ epoch {best_epoch})")
    print(f"GrC rate        : {test['grc_rate']:.4f}")
    print(f"PC rates/group  : " + "  ".join(f"{v:.4f}" for v in test["pc_rates"]))
    print(f"outputs         : {out_dir / 'best.pt'}, {csv_path}, {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
