"""Cerebellum-inspired SNN for MNIST: fixed expansion + GoC sparsity + two
opposite-polarity PC groups + inhibitory readout.

Why the architecture looks like this
------------------------------------
The biological circuit (Marr-Albus-Ito, plus the modern microzone view):

    mossy fibres (MF)  ->  granule cells (GrC)          [huge expansion, tiny fan-in]
                       ->  Golgi cells (GoC) -> GrC     [local feedback inhibition]
    GrC/parallel fibres -> Purkinje cells (PC)           [massive fan-in, the only
                                                         plastic synapse]
                       ->  molecular layer interneurons (MLI) -> PC
                                                         [FEED-FORWARD inhibition]
    climbing fibre (CF)  -> PC                          [teaching signal]
    PC -> deep cerebellar nuclei (DCN)                  [inhibitory output]
    MF collaterals -> DCN                               [excitatory drive]

Two facts drive the design:

1.  GrC weights are *not* learned.  Only PF->PC is plastic.  So the front end is a
    fixed random expansion, and the only thing training does is shape the readout.
    Sparsity is produced by GoC feedback inhibition rather than by a loss term.

2.  PC -> DCN is inhibitory and PF->PC plasticity is dominated by LTD, so a CF on
    class c can only ever *raise* score_c (weaken inhibition).  A single plastic
    site therefore cannot correct "the wrong class scored too high".  The two
    groups split that job:

        group 0 : triggered by CF present   -> LTD on the TRUE class   (release it)
        group 1 : triggered by CF absent    -> LTP on the PREDICTED class
                                               (push it down; the fnins-18 rule)

    Both groups see the SAME GrC output -- this is a "pseudo mixture of experts":
    no gating, no routing, all inputs to all groups.  Diversity comes from the
    teaching signal, not from the inputs.

Molecular layer interneurons (MLI), v4
--------------------------------------
The second cerebellar interneuron class, omitted until now.  The parallel fibres
that excite a PC also excite MLIs, and MLIs inhibit PCs: textbook *feed-forward*
inhibition, i.e. the opposite geometry to the GoC (which is feedback onto GrC).

Why multiplicative, not subtractive
    In the published conductance-based cerebellar PC model (Mitsuhashi et al.,
    Neural Networks 193:107962, 2026; table 2) the PC's GABA-A reversal is
    E_inh = -70 mV against a resting E_leak = -68 mV.  Opening g_inh = 0.70 nS on
    top of g_leak = 2.32 nS therefore moves the effective reversal to only
    -68.46 mV (a 0.46 mV hyperpolarisation) while cutting the input resistance to
    2.32/3.02 = 77%.  The MLI is a SHUNT: it divides the PC's response to
    parallel-fibre drive rather than subtracting a fixed current.  So here

        drive <- drive * (1 - clamp(mli_gain * inh, 0, 0.95))

    and NOT `drive - mli_gain * inh`.  `inh` in [0, 1] is the fraction of that
    PC's own MLI pool that spiked on this step.

Why it might matter for accuracy
    Our PCs receive only excitation plus a per-PC slow homeostatic gain (pc_h).
    Neither can express "this PC should be quiet *because the whole population is
    loud*".  Digits have very different ink mass, so the GrC code gets globally
    louder on some samples and every PC's drive rises together; the readout then
    sees a common-mode shift.  MLI is the circuit that removes it (the review we
    take the anatomy from calls the MLI drive "stable feed-forward inhibition"
    that holds PC output in place -- Nat Rev Neurosci 26:538, 2025).

Fixed, never learned
    PF->PC is the plastic site; MLI->PC is not.  All MLI weights are frozen
    buffers (non-persistent: they are a deterministic function of `seed`, so they
    never enter a checkpoint and old checkpoints keep loading unchanged).

Readout
-------
DCN integrates PC inhibition, but the PC->DCN IPSC is very brief (tau ~ 2.4 ms,
Person & Raman 2012), so inhibition only accumulates when several PCs in a
microzone fire together (measured: isolated CS -20.6%, synchronous CS -64.5%,
Tang et al. 2016).  That is a supralinear pooling inside each group:

    I[e, c] = mean_t ( mean_g p[e, c, g, t] ) ** 2          in [0, 1]

    logit_c = b_c - gamma_0 * I[0, c] - gamma_1 * I[1, c]

With G=2 a synchronous pair counts 1.0 and a lone spike 0.25 -- a 4:1 preference
for coincidence, which a plain sum would not give.

No autograd anywhere: the learning rules in `rules.py` are local and are applied
by hand.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _heav(v: torch.Tensor, thresh: float) -> torch.Tensor:
    """Heaviside spike. No surrogate needed: nothing is ever backpropagated."""
    return (v >= thresh).to(v.dtype)


def build_grc_index(n_in: int, n_grc: int, fan_in: int, seed: int = 0) -> torch.Tensor:
    """(n_grc, fan_in) int64: which MF (pixel) each granule cell samples."""
    if fan_in > n_in:
        raise ValueError(f"fan_in={fan_in} exceeds n_in={n_in}")
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.empty(n_grc, fan_in, dtype=torch.long)
    for o in range(n_grc):
        idx[o] = torch.randperm(n_in, generator=g)[:fan_in]
    return idx


class CerebellarSNN(nn.Module):
    """784 -> GrC(fixed random, GoC-sparsified) -> [MLI shunt] -> {n_group x 10 x G PCs} -> DCN."""


    def __init__(
        self,
        n_in: int = 28 * 28,
        img: int = 28,
        n_class: int = 10,
        n_grc: int = 4096,
        grc_fan_in: int = 8,
        n_goc_groups: int = 16,
        n_group: int = 1,
        n_pc_per_class: int = 1,
        # ---- v4: molecular layer interneurons (0 = disabled, exact v3 behaviour)
        n_mli: int = 0,
        mli_fan_in: int = 16,
        mli_pc_fan_in: int = 8,
        mli_in_gain: float = 4.0,
        mli_gain: float = 3.0,
        mli_beta: float = 0.9,
        mli_thresh: float = 1.0,
        T: int = 25,
        beta: float = 0.9,
        thresh: float = 1.0,
        reset: str = "subtract",
        encoder: str = "direct",
        grc_gain: float = 1.0,
        pc_gain: float = 5.0,
        goc_gain: float = 2.0,
        goc_tau: float = 2.0,
        gamma: float = 3.0,
        readout: str = "quadratic",
        mfd_dcn: str = "tonic",
        pc_homeo: bool = True,
        grc_learn: str = "off",
        seed: int = 0,
    ) -> None:
        super().__init__()
        if n_in != img * img:
            raise ValueError(f"n_in={n_in} must equal img*img={img * img}")
        if n_grc % n_goc_groups != 0:
            raise ValueError(f"n_goc_groups={n_goc_groups} must divide n_grc={n_grc}")
        if readout not in ("quadratic", "linear"):
            raise ValueError("readout must be 'quadratic' or 'linear'")
        if mfd_dcn not in ("tonic", "none", "full"):
            raise ValueError("mfd_dcn must be 'tonic', 'none' or 'full'")
        if n_mli < 0:
            raise ValueError("n_mli must be >= 0")
        if n_mli and mli_pc_fan_in > n_mli:
            raise ValueError(f"mli_pc_fan_in={mli_pc_fan_in} exceeds n_mli={n_mli}")
        if n_mli and mli_fan_in > n_grc:
            raise ValueError(f"mli_fan_in={mli_fan_in} exceeds n_grc={n_grc}")

        self.n_in, self.img, self.n_class = n_in, img, n_class
        self.n_grc, self.grc_fan_in = n_grc, grc_fan_in
        self.n_goc_groups = n_goc_groups
        self.goc_size = n_grc // n_goc_groups
        self.n_group, self.G = n_group, n_pc_per_class
        self.n_pc = n_group * n_class * n_pc_per_class
        self.n_mli = n_mli
        self.mli_fan_in, self.mli_pc_fan_in = mli_fan_in, mli_pc_fan_in
        self.mli_in_gain, self.mli_gain = mli_in_gain, mli_gain
        self.mli_beta, self.mli_thresh = mli_beta, mli_thresh
        self.mli_clamp = 0.95        # never shut a PC drive fully off
        self.T, self.beta, self.thresh = T, beta, thresh
        self.reset, self.encoder = reset, encoder
        self.grc_gain, self.pc_gain = grc_gain, pc_gain
        self.goc_gain, self.goc_tau = goc_gain, goc_tau
        # live GoC gain: a scalar buffer so a homeostatic controller can adjust it
        # when the granule layer learns and changes its own drive
        # v3: ONE GAIN PER GOC GROUP instead of a single global scalar.  A real
        # Golgi cell inhibits a local patch, so a global gain violates locality;
        # each group now adapts its own gain from its own activity.
        self.register_buffer("goc_g",
                             torch.full((n_goc_groups,), float(goc_gain)),
                             persistent=True)
        # v3: per-cell slow average of the firing rate, used by the GoC-gated
        # plasticity rule (a BCM-style sliding threshold at the MF->GrC synapse)
        self.register_buffer("grc_rate_ema", torch.zeros(n_grc), persistent=True)
        self.readout, self.mfd_dcn = readout, mfd_dcn

        # ---- MF -> GrC : FIXED, non-negative, rows sum to 1 (a weighted average
        # of its fan_in pixels, so the current is O(1) and comparable across cells)
        idx = build_grc_index(n_in, n_grc, grc_fan_in, seed)
        self.register_buffer("grc_idx", idx, persistent=True)
        w = torch.rand(n_grc, grc_fan_in, generator=torch.Generator().manual_seed(seed + 1))
        w = w / w.sum(dim=1, keepdim=True)
        self.register_buffer("grc_w", w, persistent=True)

        # ---- GrC -> MLI -> PC : FEED-FORWARD inhibition, all fixed -----------------
        # persistent=False on purpose: every one of these is a deterministic
        # function of `seed`, so they never need to ride along in a checkpoint and
        # enabling/disabling MLI cannot break loading an older checkpoint.
        if n_mli:
            self.register_buffer(
                "mli_idx", build_grc_index(n_grc, n_mli, mli_fan_in, seed + 3),
                persistent=False)
            wm = torch.rand(n_mli, mli_fan_in,
                            generator=torch.Generator().manual_seed(seed + 4))
            self.register_buffer("mli_w", wm / wm.sum(dim=1, keepdim=True),
                                 persistent=False)
            # each PC listens to its OWN sparse subset of MLIs (a sagittal zone,
            # not a whole-cortex broadcast), so different classes get different
            # amounts of shunt -> the shunt carries information, not just gain.
            g = torch.Generator().manual_seed(seed + 5)
            mli_pc = torch.zeros(self.n_pc, n_mli)
            for i in range(self.n_pc):
                sel = torch.randperm(n_mli, generator=g)[:mli_pc_fan_in]
                mli_pc[i, sel] = 1.0 / mli_pc_fan_in
            self.register_buffer("mli_pc_w", mli_pc, persistent=False)
        else:
            self.mli_idx = self.mli_w = self.mli_pc_w = None

        # ---- GrC -> PC : the only plastic weights, one set per group -------------
        # rows are kept sum-normalised to 1, so a PC's drive is the GrC activity
        # averaged under its own weight distribution.  Learning moves the
        # distribution; the normalisation stops the all-zero / all-one drift.
        w_pc = torch.rand(self.n_pc, n_grc, generator=torch.Generator().manual_seed(seed + 2))
        w_pc = w_pc / w_pc.sum(dim=1, keepdim=True)
        self.w_pc = nn.Parameter(w_pc, requires_grad=False)

        # Per-PC intrinsic gain (intrinsic plasticity / synaptic scaling).
        # Needed because the two groups are NOT symmetric: group 0 only ever gets
        # LTD (which moves w AWAY from the active GrC pattern -> its drive, and so
        # its firing rate, decays monotonically) while group 1 only ever gets LTP
        # (which moves w TOWARD it -> its rate climbs).  Without this term group 0
        # silences itself and group 1 saturates: measured after 3 batches,
        # 0.009 vs 0.302 spikes/neuron/step.  The gain keeps each PC near a common
        # operating point while the *tuning* is what learning changes.
        self.pc_homeo = pc_homeo
        self.grc_learn = grc_learn
        self.register_buffer("pc_h", torch.ones(self.n_pc), persistent=True)

        # ---- DCN ---------------------------------------------------------------
        self.b_dcn = nn.Parameter(torch.zeros(n_class), requires_grad=False)
        self.gamma = nn.Parameter(
            torch.full((n_group,), float(gamma)), requires_grad=False)
        if mfd_dcn == "full":
            self.v_dcn = nn.Parameter(
                torch.zeros(n_class, n_in).uniform_(-1e-3, 1e-3), requires_grad=False)
        else:
            self.register_buffer("v_dcn", torch.zeros(0), persistent=False)

    # ------------------------------------------------------------------ helpers ---
    def _grc_current(self, x: torch.Tensor) -> torch.Tensor:
        """Constant GrC input current for direct coding, (B, n_grc)."""
        xb = x[:, self.grc_idx]                              # (B, n_grc, fan_in)
        return (xb * self.grc_w).sum(dim=-1) * self.grc_gain

    def _reset(self, v: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if self.reset == "subtract":
            return v - self.thresh * s
        return v * (1.0 - s)

    def _pool(self, p: torch.Tensor) -> torch.Tensor:
        """Supralinear pooling of the G PCs of each (group, class) microzone.

        p: (B, n_group, n_class, G) per-step spikes -> (B, n_group, n_class)
        """
        n = p.sum(dim=-1) / self.G                            # mean fraction firing
        return n * n if self.readout == "quadratic" else n

    # ---------------------------------------------------------------- forward ---
    def forward(self, x: torch.Tensor, return_stats: bool = False):
        if x.dim() == 4:
            x = x.flatten(1)
        B = x.shape[0]
        dev, dt = x.device, x.dtype

        cur_grc = self._grc_current(x) if self.encoder == "direct" else None

        u_g = torch.zeros(B, self.n_grc, device=dev, dtype=dt)
        u_p = torch.zeros(B, self.n_pc, device=dev, dtype=dt)
        c_g = torch.zeros_like(u_g)
        c_p = torch.zeros_like(u_p)
        pool = torch.zeros(B, self.n_group, self.n_class, device=dev, dtype=dt)
        # GoC: a low-passed copy of each group's own firing rate feeds back as a
        # subtractive current.  Proportional (not integral) on purpose: an integral
        # controller cannot settle inside a single T=25 pass, and carrying state
        # across batches would make eval depend on history.
        a_lp = torch.zeros(B, self.n_goc_groups, device=dev, dtype=dt)
        goc_last = torch.zeros_like(a_lp)
        with_mli = bool(self.n_mli)
        if with_mli:
            u_m = torch.zeros(B, self.n_mli, device=dev, dtype=dt)
            c_m = torch.zeros_like(u_m)
            inh_max = torch.zeros((), device=dev, dtype=dt)
            shunt_acc = torch.zeros((), device=dev, dtype=dt)

        for _ in range(self.T):
            goc_last = self.goc_g.view(1, -1) * a_lp
            cur = cur_grc if self.encoder == "direct" else self._grc_current(torch.bernoulli(x))
            u_g = self.beta * u_g + cur - goc_last.repeat_interleave(self.goc_size, dim=1)
            s_g = _heav(u_g, self.thresh)
            u_g = self._reset(u_g, s_g)
            c_g = c_g + s_g

            drive = s_g @ self.w_pc.t()
            if with_mli:
                # PF -> MLI: the SAME GrC spikes that drive the PC also drive the
                # MLI pool, one step earlier in circuit terms -- feed-forward.
                a_m = (s_g[:, self.mli_idx] * self.mli_w).sum(dim=-1)
                u_m = self.mli_beta * u_m + self.mli_in_gain * a_m
                s_m = _heav(u_m, self.mli_thresh)
                u_m = u_m - self.mli_thresh * s_m
                c_m = c_m + s_m
                # MLI -> PC: multiplicative shunt (see the module docstring).
                # `inh` is the fraction of this PC's own MLI pool that fired.
                inh = (s_m @ self.mli_pc_w.t()).clamp_(0.0, 1.0)
                shunt = (self.mli_gain * inh).clamp_(0.0, self.mli_clamp)
                drive = drive * (1.0 - shunt)
                inh_max = torch.maximum(inh_max, inh.max())
                shunt_acc = shunt_acc + shunt.mean()
            if self.pc_homeo:
                drive = drive * self.pc_h
            u_p = self.beta * u_p + self.pc_gain * drive
            s_p = _heav(u_p, self.thresh)
            u_p = self._reset(u_p, s_p)
            c_p = c_p + s_p

            pool = pool + self._pool(s_p.view(B, self.n_group, self.n_class, self.G))

            a = s_g.view(B, self.n_goc_groups, self.goc_size).mean(dim=-1)
            a_lp = a_lp + (a - a_lp) / self.goc_tau

        I = pool / self.T                                     # (B, n_group, n_class)
        logits = -(I * self.gamma.view(1, -1, 1)).sum(dim=1)  # (B, n_class)
        if self.mfd_dcn in ("tonic", "full"):
            logits = logits + self.b_dcn
        if self.mfd_dcn == "full" and self.v_dcn.numel():
            logits = logits + x @ self.v_dcn.t()

        if not return_stats:
            return logits

        stats = {
            "grc_rate": (c_g.mean() / self.T).item(),
            "grc_rate_per_group": (c_g.view(B, self.n_goc_groups, self.goc_size)
                                   .mean(dim=0).mean(dim=-1) / self.T),  # (n_groups,)
            "grc_rate_per_neuron": c_g.mean(dim=0) / self.T,   # (n_grc,) tensor
            "grc_counts": c_g,                                 # (B, n_grc) raw counts
            "grc_counts_per_sample": c_g.sum(dim=1).mean().item(),
            "pc_rate_per_group": [
                (c_p.view(B, self.n_group, self.n_class, self.G)[:, e].mean() / self.T).item()
                for e in range(self.n_group)],
            "pc_rate_per_neuron": c_p.mean(dim=0) / self.T,     # (n_pc,)
            "pc_h_mean": self.pc_h.mean().item(),
            "I_mean_per_group": [I[:, e].mean().item() for e in range(self.n_group)],
            "I": I,                                            # (B, n_group, n_class)
            "goc_inh_mean": goc_last.mean().item(),
            "goc_inh_max": goc_last.max().item(),
            "pool_max": I.max().item(),
        }
        if with_mli:
            stats["mli_rate"] = (c_m.mean() / self.T).item()
            stats["mli_rate_per_neuron"] = c_m.mean(dim=0) / self.T   # (n_mli,)
            stats["mli_inh_max"] = inh_max.item()
            # mean fraction of PC drive removed by the MLI shunt over the pass
            stats["mli_shunt_mean"] = (shunt_acc / self.T).item()
        return logits, stats

    @torch.no_grad()
    def grc_counts(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n_grc) GrC spike counts -- the fixed feature vector, for probing."""
        if x.dim() == 4:
            x = x.flatten(1)
        B = x.shape[0]
        dev, dt = x.device, x.dtype
        cur_grc = self._grc_current(x)
        u = torch.zeros(B, self.n_grc, device=dev, dtype=dt)
        c = torch.zeros_like(u)
        a_lp = torch.zeros(B, self.n_goc_groups, device=dev, dtype=dt)
        for _ in range(self.T):
            u = self.beta * u + cur_grc - (self.goc_g.view(1, -1) * a_lp).repeat_interleave(
                self.goc_size, dim=1)
            s = _heav(u, self.thresh)
            u = self._reset(u, s)
            c = c + s
            a = s.view(B, self.n_goc_groups, self.goc_size).mean(dim=-1)
            a_lp = a_lp + (a - a_lp) / self.goc_tau
        return c

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).argmax(dim=1)

    # ----------------------------------------------------------------- counts ---
    def n_synapses(self) -> int:
        n = self.n_grc * self.grc_fan_in + self.n_pc * self.n_grc
        if self.n_mli:
            # GrC->MLI is one fan_in per MLI (NOT one per GrC), and MLI->PC is one
            # fan_in per PC (each PC listens to only its own mli_pc_fan_in MLIs)
            n += self.n_mli * self.mli_fan_in + self.n_pc * self.mli_pc_fan_in
        return n

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def n_trainable(self) -> int:
        """Weights the local rules actually modify.

        NOT the same as `n_params()`: gamma is a fixed hyperparameter that is
        stored as a Parameter only so it rides along in the checkpoint, and the
        32,768 GrC weights are frozen buffers that are not parameters at all.
        """
        n = self.w_pc.numel() + self.b_dcn.numel()
        if self.grc_learn != "off":
            n += self.grc_w.numel()      # unsupervised MF->GrC plasticity
        if self.mfd_dcn == "full" and self.v_dcn.numel():
            n += self.v_dcn.numel()
        return n

    def extra_repr(self) -> str:
        mli = (f" | MLI {self.n_mli} (fan-in {self.mli_fan_in}, shunt gain "
               f"{self.mli_gain}, {self.mli_pc_fan_in}/PC, in-gain "
               f"{self.mli_in_gain})" if self.n_mli else " | MLI off")
        return (f"784 -> GrC {self.n_grc} (fan-in {self.grc_fan_in}, GoC "
                f"{self.n_goc_groups}x{self.goc_size} lp(gain {self.goc_gain}, "
                f"tau {self.goc_tau}))"
                f" -> {self.n_group} groups x {self.n_class} x {self.G} PC "
                f"= {self.n_pc} -> DCN, T={self.T}, readout={self.readout}, "
                f"mfd_dcn={self.mfd_dcn}" + mli)


# -------------------------------------------------------------------- self-check ---
# Note: the checks below run on REAL MNIST, not on torch.rand.  Uniform random
# pixels have mean 0.5 while MNIST has 0.13, which saturates the LIF population and
# makes the activity statistics meaningless (that is exactly how the first version
# of this file fooled itself into reporting CV=0.04).
GRC_TARGET = 0.05


def _selfcheck() -> None:
    from data import get_loaders

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    net = CerebellarSNN().to(dev).eval()
    print(net)
    print(f"  synapses {net.n_synapses():,}   trainable {net.n_trainable():,}\n")

    loaders = get_loaders(batch_size=64, val_size=0, num_workers=0)
    x, y = next(iter(loaders["test"]))
    x, y = x.to(dev), y.to(dev)
    with torch.no_grad():
        logits, st = net(x, return_stats=True)

    # ---- check 1: GrC code must differentiate inputs, not sweep the ranking -----
    cnt = st["grc_rate_per_neuron"]                       # (n_grc,) spikes/step
    cv = (cnt.std() / cnt.mean().clamp_min(1e-9)).item()
    dead = (cnt == 0).float().mean().item()
    print("=== [1] GrC code shape (a rank-sweep would give CV ~ 0 and a flat "
          "rate) ===")
    print(f"  mean rate {st['grc_rate']:.4f}  CV {cv:.3f}  silent {dead * 100:.1f}%")
    q = torch.quantile(cnt, torch.tensor([0.05, 0.5, 0.95], device=dev))
    print(f"  rate p5/p50/p95: {q[0]:.4f} / {q[1]:.4f} / {q[2]:.4f}")
    print(f"  {'OK' if cv > 0.5 else '!! WARN: CV <= 0.5, code may be degenerate'}")

    # ---- check 2: is GoC actually holding activity near the target? -----------
    print(f"\n=== [2] GrC activity vs GoC design target {GRC_TARGET:.0%} ===")
    print(f"  actual {st['grc_rate']:.4f} ({st['grc_rate'] / GRC_TARGET:.2f}x target)   "
          f"GoC inh mean {st['goc_inh_mean']:.4f} max {st['goc_inh_max']:.4f}")

    # ---- check 3: every group's PCs must fire, and not saturate ----------------
    print("\n=== [3] PC firing per group (want 0.05-0.15) ===")
    for e, r in enumerate(st["pc_rate_per_group"]):
        flag = "  <-- DEAD" if r == 0 else ("  <-- HIGH" if r > 0.35 else "")
        print(f"  group {e}: {r:.4f} spikes/neuron/step{flag}")

    # ---- check 4: pooled inhibition magnitude ---------------------------------
    print("\n=== [4] pooled inhibition I[e,c] (nonzero, not saturated) ===")
    for e, v in enumerate(st["I_mean_per_group"]):
        print(f"  group {e}: mean {v:.4f}")
    print(f"  I max {st['pool_max']:.4f}   logits std {logits.std():.4f}  "
          f"range [{logits.min():+.3f}, {logits.max():+.3f}]")

    # ---- check 5: learning moves weights but keeps the row sums ----------------
    print("\n=== [5] normalised update keeps row sums at 1 ===")
    from rules import CaveConfig, cave_update
    net.train()
    with torch.no_grad():
        z = st["grc_counts"] / net.T
        before = net.w_pc.sum(dim=1).clone()
        diag = cave_update(net, z, logits.detach(), y, CaveConfig())
        after = net.w_pc.sum(dim=1)
    print(f"  row-sum before {before.min():.6f}..{before.max():.6f}   "
          f"after {after.min():.6f}..{after.max():.6f}")
    print(f"  |dW| group0 {diag['dW0_abs']:.3e}  group1 {diag['dW1_abs']:.3e}   "
          f"wrong {diag['n_wrong']}/{x.shape[0]}")
    ok = abs(after.min() - 1) < 1e-3 and abs(after.max() - 1) < 1e-3
    print(f"  {'OK' if ok else '!! row sums drifted'}")

    # ---- check 6: the bidirectional rule must move the two rows OPPOSITE ways ---
    print("\n=== [6] sign check: w[true] must move AWAY from the active GrC pattern "
          "(LTD),\n       w[predicted] must move TOWARD it (LTP) -- same matrix ===")
    net2 = CerebellarSNN().to(dev).eval()
    with torch.no_grad():
        lg2, st2 = net2(x, return_stats=True)
        z = st2["grc_counts"] / net2.T
        w_before = net2.w_pc.detach().clone()
        pred = lg2.argmax(1)
        cave_update(net2, z, lg2, y, CaveConfig())
        dw = net2.w_pc.detach() - w_before
        G = net2.G
        y0, d0 = int(y[0]), int(pred[0])
        # rows are renormalised to sum 1, so the ROW SUM of dw is zero by
        # construction; the meaningful quantity is the overlap with this sample's
        # GrC pattern
        z0 = z[0]
        ov_y = (dw[y0 * G] * z0).sum().item()
        ov_d = (dw[d0 * G] * z0).sum().item()
        print(f"  sample 0: y={y0}  pred={d0}  (wrong={y0 != d0})")
        print(f"  <dw[true], z>      = {ov_y:+.3e}   (expect < 0: move away, LTD)")
        print(f"  <dw[predicted], z> = {ov_d:+.3e}   (expect > 0: move toward, LTP)")
        print(f"  {'OK' if ov_y < 0 < ov_d else '!! SIGN ERROR'}")

    # ---- check 7: MLI must shunt the PC drive and must NOT touch the encoder ----
    print("\n=== [7] MLI (PF->MLI->PC feed-forward shunt, v4) ===")
    net3 = CerebellarSNN(n_mli=20).to(dev).eval()
    with torch.no_grad():
        lg3, st3 = net3(x, return_stats=True)
    same_grc = torch.equal(st3["grc_counts"], st["grc_counts"])
    dlog = (lg3 - logits).abs().mean().item()
    print(f"  MLI rate {st3['mli_rate']:.4f}  shunt mean {st3['mli_shunt_mean']:.3f}"
          f"  per-PC max {st3['mli_inh_max']:.3f}")
    print(f"  GrC counts identical to the no-MLI run : {same_grc}   "
          f"(MLI must not alter the encoder)")
    print(f"  mean |dlogit| vs the no-MLI model      : {dlog:.4f}   "
          f"(0 => the shunt is dead)")
    ok7 = same_grc and st3["mli_rate"] > 1e-4 and 0.02 < st3["mli_shunt_mean"] < 0.9
    print(f"  {'OK' if ok7 else '!! MLI is dead, saturated, or leaking into GrC'}")


if __name__ == "__main__":
    _selfcheck()
