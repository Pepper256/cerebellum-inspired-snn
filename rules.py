"""Local (non-backprop) learning rules for the cerebellum-inspired SNN.

There is no autograd anywhere.  Every update is a three-factor-style product of

    (post-synaptic eligibility) x (pre-synaptic activity) x (climbing-fibre error)

which for our purposes collapses to a delta rule on the FIXED GrC feature vector:

    dW[e, c] = +/- eta_e * (error indicator for class c) (x) z

where z_j = (GrC_j spike count)/T is the normalised pre-synaptic activity.

Sign logic (the part that is easy to get backwards)
---------------------------------------------------
PC -> DCN is inhibitory, so

    score_c = b_c - gamma * I_c,      I_c increases with w_c

i.e. **strengthening a PC lowers that class's score**, and a CF on class c can
therefore only ever *raise* score_c.  Two groups are needed:

    group 0  CF PRESENT  -> LTD on the TRUE class       (weaken inhibition -> raise
                                                         the correct score)
    group 1  CF ABSENT   -> LTP on the PREDICTED class  (strengthen inhibition ->
                                                         push the wrong score down)

Group 1's rule is the LTP branch of the published cerebellar rule in
Shinji, Okuno & Hirata, Front. Neurosci. 18:1220908 (2024):

    dw/dt = -gamma_LTD * s_CF * q_GrC + gamma_LTP * (NOT s_CF) * s_GrC

Updates are applied only on misclassified samples; a correct prediction leaves the
network untouched (which also keeps the groups from saturating).

Row normalisation (sum = 1, i.e. homeostatic synaptic scaling) is not cosmetic: it
turns the LTD into a *contrastive* rule -- depressing w_y on class-y patterns and
renormalising is equivalent to raising it everywhere else.  The NIHMS study
normalises predicted LTD for exactly this reason ("to maintain overall PkC input
strength, as likely achieved biologically by homeostatic and opposing processes
like LTP").
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

EPS = 1e-12


@dataclass
class CaveConfig:
    eta0: float = 0.10          # group 0 LTD rate
    eta1: float = 0.10          # group 1 LTP rate
    eta_b: float = 0.01         # DCN bias rate
    eta_v: float = 0.01         # MF->DCN (excitatory path) rate, only if mfd_dcn="full"
    group1_scope: str = "argmax"   # "argmax" | "all"
    normalise: bool = True
    use_cf: bool = True         # ablation a: turn the teaching signal off
    # ---- intrinsic plasticity: keep every PC near a common operating point ----
    # Without it group 0 (LTD only) decays to silence and group 1 (LTP only)
    # saturates.  This is a slow multiplicative controller on each PC's gain, which
    # is exactly what intrinsic plasticity / synaptic scaling does.
    pc_homeo: bool = True
    eta_h: float = 0.5
    rate_target: float = 0.10
    h_min: float = 0.05
    h_max: float = 20.0
    # must match the CLI default in train.py: "both"/"g0"/"g1" are the ruled-out
    # two-separate-matrix variants and are only reachable on purpose
    groups: str = "bidir"
    # ---- unsupervised GrC plasticity (granule layer, NO teaching signal) ------
    grc_learn: str = "off"      # "off" | "hebb"
    eta_g: float = 0.02
    # ---- homeostatic control of the GoC gain -------------------------------
    # Needed once the granule layer learns: the Hebbian rule concentrates each
    # row onto fewer afferents, which raises the drive (measured 4.1% -> 13%)
    # and breaks the operating point the PC readout was calibrated for.
    goc_homeo: bool = False
    eta_goc: float = 0.5
    goc_target: float = 0.05
    # ---- v3: GoC gates the DIRECTION of MF->GrC plasticity (BCM-style) -----
    goc_gate: bool = False
    gate_kappa: float = 1.0
    gate_scale: float = 0.02
    ema_alpha: float = 0.1


def _renorm(w: torch.Tensor) -> None:
    w.clamp_(min=0.0)
    w.div_(w.sum(dim=1, keepdim=True).clamp_min(EPS))


@torch.no_grad()
def cave_update(
    model,
    z: torch.Tensor,
    logits: torch.Tensor,
    y: torch.Tensor,
    cfg: CaveConfig,
    x: torch.Tensor | None = None,
    pc_rate: torch.Tensor | None = None,
) -> dict:
    """One local update.

    z       : (B, n_grc) normalised GrC activity (counts / T)
    logits  : (B, n_class) produced by the forward pass
    y       : (B,) labels
    pc_rate : (n_pc,) this batch's per-PC firing rate, for intrinsic plasticity
    """
    B = y.shape[0]
    n_cls, G = model.n_class, model.G
    pred = logits.argmax(dim=1)
    wrong = pred != y
    n_wrong = int(wrong.sum())
    diag = {"n_wrong": n_wrong, "dW0_abs": 0.0, "dW1_abs": 0.0, "h_range": (1.0, 1.0)}

    # ---- intrinsic plasticity: runs on EVERY batch, independent of the error ----
    # (it is a controller on excitability, not a learning signal).  Ratio form
    # rather than incremental: gain ~ drive ~ rate, so the ratio is close to the
    # fixed point in one step and the 0.5 exponent damps any overshoot.
    if cfg.pc_homeo and pc_rate is not None and getattr(model, "pc_homeo", False):
        ratio = cfg.rate_target / pc_rate.clamp_min(1e-4)
        model.pc_h.mul_(ratio.pow(cfg.eta_h)).clamp_(cfg.h_min, cfg.h_max)
        diag["h_range"] = (model.pc_h.min().item(), model.pc_h.max().item())

    if n_wrong == 0 or not cfg.use_cf:
        return diag

    # ---- group 0: fire on the TRUE class of the misclassified samples ----------
    err0 = torch.zeros(B, n_cls, device=z.device, dtype=z.dtype)
    err0[torch.arange(B, device=z.device), y] = wrong.to(z.dtype)

    # ---- group 1: fire on the PREDICTED class (it was silent on the true one) --
    err1 = torch.zeros(B, n_cls, device=z.device, dtype=z.dtype)
    if cfg.group1_scope == "argmax":
        err1[torch.arange(B, device=z.device), pred] = wrong.to(z.dtype)
    elif cfg.group1_scope == "all":
        err1.copy_(wrong.to(z.dtype).unsqueeze(1).expand(B, n_cls))
        err1[torch.arange(B, device=z.device), y] = 0.0
    else:
        raise ValueError(f"unknown group1_scope {cfg.group1_scope!r}")

    # ---- eta is DIMENSIONLESS: normalise each sample's GrC pattern to L1 = 1, so
    # "eta" means "fraction of the row's total weight mass moved", independent of
    # n_grc.  Without this the raw counts (mean 0.04, entries 2.4e-4 after the
    # sum-to-1 normalisation) make a nominal eta=0.1 an update ~16x larger than
    # the weights themselves -- which is what drove group 0 to silence.
    zn = z / z.sum(dim=1, keepdim=True).clamp_min(EPS)

    with torch.no_grad():
        if cfg.groups in ("bidir", "bidir_all"):
            # ONE matrix receives BOTH directions.  This is the perceptron rule in
            # inhibitory clothing and it is the only variant with a fixed point:
            # with two separate matrices (LTD-only / LTP-only) neither has a
            # restoring force, so both drift monotonically -- measured: group 0
            # decayed to silence, and a 15-epoch run peaked at epoch 3 (46.6%) then
            # fell to 42% while group 0's rows drifted toward uniform.
            #
            #   c == y  and wrong  -> class needs a HIGHER score -> weaken w_c (LTD)
            #   c == yhat and wrong -> class needs a LOWER score -> strengthen w_c
            #
            # "bidir" trains only the first microzone; "bidir_all" applies the same
            # rule to every microzone (a true ensemble of identically-wired copies,
            # which is the only reading of "grouping" not yet ruled out).
            g_ltd = (err0.t() @ zn) / B
            g_ltp = (err1.t() @ zn) / B
            n_zone = model.n_group if cfg.groups == "bidir_all" else 1
            for e in range(n_zone):
                w = model.w_pc[e * n_cls * G:(e + 1) * n_cls * G]
                w.sub_(cfg.eta0 * g_ltd.repeat_interleave(G, dim=0))
                w.add_(cfg.eta1 * g_ltp.repeat_interleave(G, dim=0))
                if cfg.normalise:
                    _renorm(w)
            diag["dW0_abs"] = (cfg.eta0 * g_ltd).abs().max().item()
            diag["dW1_abs"] = (cfg.eta1 * g_ltp).abs().max().item()

        if cfg.groups in ("both", "g0"):
            g0 = (err0.t() @ zn) / B                      # (n_cls, n_grc)
            g0 = g0.repeat_interleave(G, dim=0)           # one row per PC in the microzone
            w0 = model.w_pc[: n_cls * G]
            w0.sub_(cfg.eta0 * g0)                        # LTD
            if cfg.normalise:
                _renorm(w0)
            diag["dW0_abs"] = (cfg.eta0 * g0).abs().max().item()

        if cfg.groups in ("both", "g1"):
            g1 = (err1.t() @ zn) / B
            g1 = g1.repeat_interleave(G, dim=0)
            w1 = model.w_pc[n_cls * G:]
            w1.add_(cfg.eta1 * g1)                        # LTP
            if cfg.normalise:
                _renorm(w1)
            diag["dW1_abs"] = (cfg.eta1 * g1).abs().max().item()

        # ---- DCN bias: raise the true class, lower the predicted one -----------
        if model.mfd_dcn in ("tonic", "full"):
            d_y = torch.zeros(n_cls, device=z.device, dtype=z.dtype)
            d_y.index_add_(0, y[wrong], torch.ones(n_wrong, device=z.device, dtype=z.dtype))
            d_d = torch.zeros(n_cls, device=z.device, dtype=z.dtype)
            d_d.index_add_(0, pred[wrong], torch.ones(n_wrong, device=z.device, dtype=z.dtype))
            model.b_dcn.add_(cfg.eta_b * (d_y - d_d) / B)

        # ---- optional excitatory MF->DCN path: SAME direction, OPPOSITE sign ----
        # because it enters the logit with a '+' instead of a '-'
        if model.mfd_dcn == "full" and model.v_dcn.numel() and x is not None:
            xh = x.reshape(B, -1) / model.n_in
            acc = torch.zeros_like(model.v_dcn)
            acc.index_add_(0, y[wrong], xh[wrong])
            acc.index_add_(0, pred[wrong], -xh[wrong])
            model.v_dcn.add_(cfg.eta_v * acc / B)

    return diag

@torch.no_grad()
def grc_update(
    model,
    x: torch.Tensor,
    grc_counts: torch.Tensor,
    cfg: CaveConfig,
) -> dict:
    """UNSUPERVISED plasticity at the MF->GrC synapse.  No labels, no CF.

    Biological rationale: the granule layer is an unsupervised sparse-coding
    stage (Marr-Albus expansion + Golgi inhibition); only the parallel-fibre ->
    Purkinje synapse is taught by the climbing fibre.  So the natural hybrid is
    "unsupervised front end + supervised readout", the recipe also used by
    Diehl & Cook and by the cerebellum-inspired classifier of Zhou et al.

    The MF->GrC weights are SPARSE-SAMPLED: GrC j owns only its `fan_in` pixels
    and its row sums to 1, so the consistent target for a Hebbian step is the
    RELATIVE pattern among that cell's own afferents:

        xhat[b, j, i] = x[b, idx[j, i]] / sum_i' x[b, idx[j, i']]

    and the rule is a competitive-learning / Oja-style average over the samples
    that MAKE THIS CELL FIRE:

        dw[j] = eta_g * ( sum_b s[b,j] xhat[b,j] - (sum_b s[b,j]) w[j] ) / B

    Only x, s and w are used: fully local, no label.

    Collapse guard: a cell that stops firing stops updating (sum_b s = 0), so a
    dead unit cannot be revived by this rule alone -- the GoC loop has to keep
    activity up.  `grc_row_max_mean` therefore doubles as a collapse detector:
    it approaches 1.0 if every row collapses onto a single afferent.
    """
    if cfg.grc_learn != "hebb":
        return {"grc_dW": 0.0, "grc_row_max_mean": float("nan"),
                "grc_rows_updated": 0}

    B = x.shape[0]
    xf = x.reshape(B, -1)
    xb = xf[:, model.grc_idx]                     # (B, n_grc, fan_in)
    tot = xb.sum(dim=-1, keepdim=True)            # (B, n_grc, 1)
    valid = tot.squeeze(-1) > 1e-8                # target defined only if sum > 0
    xb.div_(tot.clamp_min(EPS))                   # in place: rows sum to 1

    s = grc_counts * valid.to(grc_counts.dtype)   # update only where the target exists
    num = torch.einsum("bj,bjm->jm", s, xb)       # (n_grc, fan_in)
    den = s.sum(dim=0).unsqueeze(1)               # (n_grc, 1)

    w = model.grc_w
    direction = num - den * w                     # Oja-style: points at the input

    if cfg.goc_gate:
        # ---- v3: GoC gates the DIRECTION of MF->GrC plasticity ------------------
        # Roggeri et al. (J Neurosci): with the Golgi circuit intact LTD prevails
        # at MF->GrC; blocking GABA-A (gabazine) flips it to LTP.  So the state of
        # inhibition -- not just "did this cell fire" -- decides the sign.
        #
        # Implemented as a BCM-style sliding threshold on the CELL'S OWN rate:
        #   over-active cell -> gate -> -1  (LTD, weaken what drives it)
        #   quiet cell       -> gate -> +1  (LTP, sensitise it)
        # factor = 1 + kappa * gate; kappa <= 1 only modulates, kappa > 1 can flip
        # the sign of the update.
        rate = grc_counts.mean(dim=0) / max(model.T, 1)          # (n_grc,)
        model.grc_rate_ema.mul_(1.0 - cfg.ema_alpha).add_(cfg.ema_alpha * rate)
        gate = torch.tanh((cfg.goc_target - model.grc_rate_ema) / max(cfg.gate_scale, 1e-6))
        factor = (1.0 + cfg.gate_kappa * gate).unsqueeze(1)      # (n_grc, 1)
        direction = direction * factor

    dW = cfg.eta_g * direction / B

    w.add_(dW).clamp_(min=0.0)
    w.div_(w.sum(dim=1, keepdim=True).clamp_min(EPS))          # keep rows on the simplex

    out = {
        "grc_dW": dW.abs().max().item(),
        "grc_row_max_mean": w.max(dim=1).values.mean().item(),
        "grc_rows_updated": int((den > 0).sum()),
    }
    if cfg.goc_gate:
        out["grc_gate_mean"] = float(gate.mean())
        out["grc_gate_neg_frac"] = float((gate < 0).float().mean())
    return out


@torch.no_grad()
def goc_homeo_update(model, grc_rate_per_group: torch.Tensor, cfg: CaveConfig) -> dict:
    """Per-group GoC gain control (v3).

    Each of the `n_goc_groups` groups owns one gain, adjusted from ITS OWN firing
    rate -- no global scalar.  This keeps the rule local and lets different
    patches sit at different operating points, as real Golgi cells do.

    NEGATIVE feedback: too active -> raise that group's gain; too quiet -> lower.
    Multiplicative, with the per-step change soft-clipped to [0.5, 2] so one bad
    batch cannot blow the loop up.
    """
    if not cfg.goc_homeo:
        return {"goc_g_mean": float(model.goc_g.mean())}
    ratio = (grc_rate_per_group / max(cfg.goc_target, 1e-4)).clamp(0.5, 2.0)
    model.goc_g.mul_(ratio.pow(cfg.eta_goc)).clamp_(0.2, 200.0)
    return {"goc_g_mean": float(model.goc_g.mean()),
            "goc_g_min": float(model.goc_g.min()),
            "goc_g_max": float(model.goc_g.max())}
