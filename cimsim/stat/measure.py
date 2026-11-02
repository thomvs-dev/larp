"""A3 step 1: measure the output error of bit-accurate CIM layers.

The error is taken in integer MAC units, before scaling:

    e[t, n] = y_bit_accurate[t, n] - (qx @ qw.T)[t, n]

so it describes the macro (ADC rounding, clipping, noise) independently of the activation and
weight scales. The statistical mode (A3 step 2) injects a model of e into the exact integer matmul.

Usage:
    probe = ErrorProbe()
    probe.attach(model)          # every bit_accurate CIM layer
    model(...)                   # bit-accurate forward passes; errors propagate as in a real run
    stats = probe.summary()      # per-layer statistics (JSON-serialisable)
    samples = probe.samples()    # per-layer tensors for model fitting
    probe.detach(model)
"""

from __future__ import annotations

import math

import torch


def _quantiles(x: torch.Tensor, qs=(0.1, 0.5, 0.9, 1.0)) -> list[float]:
    x = x.double().flatten()
    return [float(torch.quantile(x, q)) for q in qs]


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    d = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum()) / d if d > 0 else float("nan")


class _LayerAcc:
    def __init__(self, n_out: int, k_in: int, max_samples: int, gen: torch.Generator) -> None:
        self.N, self.K = n_out, k_in
        self.n = 0
        self.s1 = self.s2 = self.s3 = self.s4 = 0.0      # raw moments of e
        self.ideal2 = 0.0
        self.n_zero = 0
        self.max_abs = 0.0
        self.ch_s1 = torch.zeros(n_out, dtype=torch.float64)
        self.ch_s2 = torch.zeros(n_out, dtype=torch.float64)
        self.tok: dict[str, list[torch.Tensor]] = {k: [] for k in
                                                    ("e_ms", "ideal_ms", "qx_absmean", "qx_neg_frac", "e_mean")}
        self.samp_ideal: list[torch.Tensor] = []
        self.samp_e: list[torch.Tensor] = []
        self.n_samp = 0
        self.max_samples = max_samples
        self.gen = gen

    def add(self, qx: torch.Tensor, qw: torch.Tensor, y: torch.Tensor) -> None:
        ideal = qx.double() @ qw.double().t()                       # exact in float64 for 8-bit operands
        e = y.double() - ideal
        # float32 output of the executors: values are integers held in float32; allow its rounding
        tol = 0.5 + ideal.abs() * 2.0 ** -23
        self.n += e.numel()
        self.s1 += float(e.sum())
        self.s2 += float((e * e).sum())
        self.s3 += float((e ** 3).sum())
        self.s4 += float((e ** 4).sum())
        self.ideal2 += float((ideal * ideal).sum())
        self.n_zero += int((e.abs() <= tol).sum())
        self.max_abs = max(self.max_abs, float(e.abs().max()))
        self.ch_s1 += e.sum(0)
        self.ch_s2 += (e * e).sum(0)
        q = qx.double()
        self.tok["e_ms"].append((e * e).mean(1).float())
        self.tok["e_mean"].append(e.mean(1).float())
        self.tok["ideal_ms"].append((ideal * ideal).mean(1).float())
        self.tok["qx_absmean"].append(q.abs().mean(1).float())
        self.tok["qx_neg_frac"].append((q < 0).double().mean(1).float())
        room = self.max_samples - self.n_samp
        if room > 0:
            k = min(room, max(1, self.max_samples // 16), e.numel())
            idx = torch.randperm(e.numel(), generator=self.gen)[:k]
            self.samp_ideal.append(ideal.flatten()[idx].float())
            self.samp_e.append(e.flatten()[idx].float())
            self.n_samp += k

    def summary(self, n_bins: int = 8) -> dict:
        n = max(self.n, 1)
        m1, m2 = self.s1 / n, self.s2 / n
        var = max(m2 - m1 * m1, 0.0)
        std = math.sqrt(var)
        m3c = self.s3 / n - 3 * m1 * m2 + 2 * m1 ** 3
        m4c = self.s4 / n - 4 * m1 * self.s3 / n + 6 * m1 * m1 * m2 - 3 * m1 ** 4
        ch_cnt = n / self.N
        ch_mean = self.ch_s1 / ch_cnt
        ch_std = (self.ch_s2 / ch_cnt - ch_mean ** 2).clamp_min(0).sqrt()
        tok = {k: torch.cat(v) for k, v in self.tok.items()}
        out = {
            "K": self.K, "N": self.N, "tokens": int(tok["e_ms"].numel()), "elements": self.n,
            "rel_err": math.sqrt(self.s2 / self.ideal2) if self.ideal2 > 0 else float("nan"),
            "mean": m1, "std": std,
            "bias_over_std": m1 / std if std > 0 else 0.0,
            "skew": m3c / std ** 3 if std > 0 else 0.0,
            "excess_kurtosis": m4c / var ** 2 - 3.0 if var > 0 else 0.0,
            "frac_exact": self.n_zero / n,
            "max_abs": self.max_abs,
            "ideal_rms": math.sqrt(self.ideal2 / n),
            # spread of the error std across output channels (1.0 = all channels equal)
            "channel_std_q10_q50_q90_max": [v / std if std > 0 else 0.0 for v in _quantiles(ch_std)],
            "channel_mean_over_std_q10_q50_q90_max": _quantiles(ch_mean / std) if std > 0 else [0.0] * 4,
            # spread of the per-token RMS error, relative to the layer RMS error
            "token_rms_q10_q50_q90_max": [v / math.sqrt(m2) if m2 > 0 else 0.0
                                          for v in _quantiles(tok["e_ms"].sqrt())],
            # what predicts the per-token error size
            "corr_log_token_ms__log_ideal_ms": _corr(tok["e_ms"].clamp_min(1e-12).log(),
                                                     tok["ideal_ms"].clamp_min(1e-12).log()),
            "corr_token_ms__qx_absmean": _corr(tok["e_ms"], tok["qx_absmean"]),
            "corr_token_ms__qx_neg_frac": _corr(tok["e_ms"], tok["qx_neg_frac"]),
        }
        # error std as a function of |ideal| (quantile bins of the sampled elements)
        if self.samp_e:
            si, se = torch.cat(self.samp_ideal).double(), torch.cat(self.samp_e).double()
            a = si.abs()
            edges = torch.quantile(a, torch.linspace(0, 1, n_bins + 1, dtype=torch.float64))
            rows = []
            for b in range(n_bins):
                m = (a >= edges[b]) & ((a < edges[b + 1]) if b < n_bins - 1 else (a <= edges[b + 1]))
                if m.sum() > 1:
                    rows.append({"abs_ideal_lo": float(edges[b]), "abs_ideal_hi": float(edges[b + 1]),
                                 "e_mean": float(se[m].mean()), "e_std": float(se[m].std())})
            out["std_vs_abs_ideal"] = rows
        return out

    def samples(self) -> dict:
        tok = {k: torch.cat(v) for k, v in self.tok.items()}
        return {"ideal": torch.cat(self.samp_ideal) if self.samp_ideal else torch.empty(0),
                "e": torch.cat(self.samp_e) if self.samp_e else torch.empty(0), **{f"tok_{k}": v for k, v in tok.items()},
                "ch_mean": (self.ch_s1 / max(self.n / self.N, 1)).float(),
                "ch_ms": (self.ch_s2 / max(self.n / self.N, 1)).float()}


class ErrorProbe:
    def __init__(self, max_samples_per_layer: int = 20000, seed: int = 0) -> None:
        self.max_samples = max_samples_per_layer
        self.gen = torch.Generator().manual_seed(seed)
        self.layers: dict[str, _LayerAcc] = {}
        self.meta: dict[str, dict] = {}

    def __call__(self, layer, qx: torch.Tensor, qw: torch.Tensor, y: torch.Tensor) -> None:
        acc = self.layers.get(layer.name)
        if acc is None:
            acc = self.layers[layer.name] = _LayerAcc(qw.shape[0], qw.shape[1], self.max_samples, self.gen)
            c = layer.cfg
            self.meta[layer.name] = {"nrow": c.nrow, "adc_bits": c.adc_bits, "act_enc": c.act_enc,
                                     "adc_range": c.adc_range, "rand_noise_sigma": c.rand_noise_sigma,
                                     "non_linear_sigma": c.non_linear_sigma,
                                     "adc_fs": None if layer.adc_fs is None else layer.adc_fs.tolist()}
        acc.add(qx.cpu(), qw.cpu(), y.cpu())

    def attach(self, model) -> list[str]:
        names = []
        for m in model.modules():
            if hasattr(m, "_probe") and getattr(m, "cfg", None) is not None and m.cfg.mode == "bit_accurate":
                m._probe = self
                names.append(m.name)
        return names

    @staticmethod
    def detach(model) -> None:
        for m in model.modules():
            if hasattr(m, "_probe"):
                m._probe = None

    def summary(self) -> dict:
        return {name: {**self.meta[name], **acc.summary()} for name, acc in self.layers.items()}

    def samples(self) -> dict:
        return {name: acc.samples() for name, acc in self.layers.items()}
