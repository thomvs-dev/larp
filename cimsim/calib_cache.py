"""Cache of calibration MAC histograms (A5).

The calibration pass (cimsim.calibrate.calibrate_adc_range) runs the model with every CIM layer in
ideal mode and records, per calibrated layer, the histogram of ideal MAC values per (cycle, plane) and
the largest MAC per output column. ADC full scale and the statistical error model are then computed
from these in a few seconds; the forward pass that records them takes 98 % of the calibration time
(35–85 min on the result models). The histograms do not depend on ADC bits, ADC range method or
granularity, analog noise, hybrid levels, seed or backend, so sweeps over those reuse them.

What a layer's histogram depends on, and therefore what the cache key holds:

- model-level key (directory): model weights (per-parameter float64 sum and absolute sum), calibration
  data (caller-supplied `data_key`, e.g. a hash of the token ids and the window length), torch and
  transformers versions, and, for every CIM layer, the settings that change its ideal output (fp or
  quantized, bit widths, granularities, signedness, weight format, KV granularity). In ideal mode a
  layer outputs q_x · q_wᵀ, which does not depend on nrow or encoding, so those are per-layer only.
- code: hash of the source files that compute quantization, ideal outputs and histograms
  (HIST_SOURCES); any edit there starts a new cache.
- per-layer key (file): layer name, every config field except the ones listed in HIST_INDEPENDENT,
  mode reduced to fp / quantized, row chunks (conv layers), weight shape.

A layer with a cache hit is not recorded again; if every calibrated layer hits, the forward pass is
skipped. Cached and fresh calibration give identical ADC full scales and statistical parameters
(tests/test_calib_cache.py).

Location: `calib_cache/` in the repository (git-ignored), or $CIMSIM_CALIB_CACHE; "off" disables it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn as nn

FORMAT_VERSION = 1

# source files whose code decides the ideal outputs or the histograms; a change in any of them
# invalidates the cache (a bug fix in quantization must not be hidden by old histograms)
HIST_SOURCES = ("cimsim/config.py", "cimsim/quant.py", "cimsim/core/bitplanes.py", "cimsim/core/calib.py",
                "cimsim/core/layers.py", "cimsim/core/attention.py", "cimsim/models/swap.py",
                "cimsim/calibrate.py", "cimsim/calib_cache.py")

# config fields that do not change the calibration histogram of a layer
HIST_INDEPENDENT = {
    "adc_bits", "adc_range", "adc_range_gran", "adc_range_method", "adc_range_pct",
    "rand_noise_sigma", "non_linear_sigma", "hybrid_levels", "token_block", "seed", "backend",
    "stat_saturation", "stat_noise", "stat_flip_lambda_max", "stat_flip_kurtosis_min", "stat_sat_margin",
    "mode", "macro",                  # replaced by "fp" and "calibrated" below
}
# fields that change a layer's ideal output (so the inputs of later layers)
IDEAL_OUTPUT_FIELDS = ("w_bits", "x_bits", "w_gran", "x_gran", "signed_act", "w_quant", "kv_gran")

_REPO = Path(__file__).resolve().parents[1]


def default_cache_dir() -> Path | None:
    env = os.environ.get("CIMSIM_CALIB_CACHE")
    if env is not None:
        return None if env.strip().lower() in ("", "off", "none", "0") else Path(env)
    return _REPO / "calib_cache"


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def tensor_key(t: torch.Tensor) -> str:
    """Hash of a tensor's contents (for calibration token ids)."""
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def source_fingerprint() -> str:
    h = hashlib.sha256()
    for rel in HIST_SOURCES:
        h.update(rel.encode())
        h.update((_REPO / rel).read_bytes().replace(b"\r\n", b"\n"))   # same hash on Windows checkouts
    return h.hexdigest()[:16]


def weight_fingerprint(model: nn.Module) -> list:
    out = []
    with torch.no_grad():
        for name, p in model.named_parameters():
            out.append([name, list(p.shape), str(p.dtype),
                        float(torch.sum(p, dtype=torch.float64)),
                        float(torch.sum(p.abs(), dtype=torch.float64))])
    return out


def _cfg_dict(cfg) -> dict:
    return {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}


def model_key(model: nn.Module, cim_layers: list, data_key: str, needs_calibration) -> dict:
    import transformers
    layers = []
    for m in cim_layers:
        d = _cfg_dict(m.cfg)
        layers.append([m.name, type(m).__name__, d["mode"] == "fp"] + [d[f] for f in IDEAL_OUTPUT_FIELDS])
    return {"format": FORMAT_VERSION, "source": source_fingerprint(), "data": data_key, "torch": torch.__version__,
            "transformers": transformers.__version__, "weights": _sha(weight_fingerprint(model)),
            "layers": layers}


def layer_key(m, needs_calibration) -> dict:
    d = {k: v for k, v in _cfg_dict(m.cfg).items() if k not in HIST_INDEPENDENT}
    d["fp"] = m.cfg.mode == "fp"
    d["calibrated"] = needs_calibration(m.cfg)
    d["name"] = m.name
    d["type"] = type(m).__name__
    if not m._dynamic_operand:
        d["shape"] = list(m._weight_matrix().shape)
        ch = m._chunks()
        d["chunks"] = [list(c) for c in ch] if ch is not None else None
    return d


class CalibCache:
    def __init__(self, root: Path, mkey: dict) -> None:
        self.dir = Path(root) / _sha(mkey)[:20]
        self.mkey = mkey

    def _file(self, lkey: dict) -> Path:
        return self.dir / f"{_sha(lkey)[:24]}.pt"

    def load(self, m, lkey: dict) -> bool:
        f = self._file(lkey)
        if not f.exists():
            return False
        e = torch.load(f, map_location="cpu", weights_only=True)
        if e.get("layer_key") != json.dumps(lkey, sort_keys=True, default=str):
            return False                                   # hash collision or stale file: recompute
        m._hist = e["hist"]
        m._cal_colmax = e["colmax"].to(torch.int64) if e.get("colmax") is not None else None
        if e.get("cal_K") is not None:
            m._cal_K = int(e["cal_K"])
        return True

    def save(self, m, lkey: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        meta = self.dir / "model_key.json"
        if not meta.exists():
            meta.write_text(json.dumps(self.mkey, indent=1, default=str), encoding="utf-8")
        cm = m._cal_colmax
        if cm is not None:                                 # integer MAC maxima: store compactly
            cm = cm.to(torch.int16 if float(cm.max()) < 32768 else torch.int32)
        e = {"layer_key": json.dumps(lkey, sort_keys=True, default=str), "hist": m._hist,
             "colmax": cm, "cal_K": getattr(m, "_cal_K", None)}
        tmp = self._file(lkey).with_suffix(".tmp")
        torch.save(e, tmp)
        os.replace(tmp, self._file(lkey))
