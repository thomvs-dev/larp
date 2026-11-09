"""Shared setup for the run scripts: load a HuggingFace model, swap in CIM layers,
calibrate the ADC range if any layer needs it."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import torch
import yaml

from cimsim.calibrate import calibrate_adc_range, needs_calibration
from cimsim.config import LayerConfigMap
from cimsim.events import EventCounter
from cimsim.models.swap import swap_linears
from eval.perplexity import load_wikitext2_ids

ROOT = Path(__file__).resolve().parents[1]


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def load_model(name: str, device: str, dtype_name: str = "float32"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, dtype_name)
    if device == "cpu" and dtype == torch.float16:
        dtype = torch.float32
    tok = AutoTokenizer.from_pretrained(name)
    try:  # transformers >= 4.56 uses `dtype`; older versions use `torch_dtype`
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    return model.to(device), tok, dtype


def swap_from_config(model, config_path: str, counter: EventCounter) -> tuple[dict | None, list[str]]:
    """Returns (config dict or None, replaced layer names). config_path 'none' leaves the model unchanged."""
    if config_path.lower() == "none":
        return None, []
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    cfg_map = LayerConfigMap.from_dict(cfg_dict)
    replaced = swap_linears(model, cfg_map, counter)
    if cfg_dict.get("attention", False):            # QKᵀ and P·V on CIM arrays (cimsim.core.attention)
        from cimsim.core.attention import swap_attention
        replaced += swap_attention(model, cfg_map, counter)
    return cfg_dict, replaced


def context_len(model) -> int:
    return getattr(model.config, "max_position_embeddings", None) or getattr(model.config, "n_positions", 2048)


_DEFAULT = object()


def calibrate_if_needed(model, tok, seq_len: int, calib_tokens: int, device: str, cache_dir=_DEFAULT,
                        info: dict | None = None) -> tuple[list[str], float]:
    """Calibrate the ADC range on the first `calib_tokens` WikiText-2 train tokens if any layer uses
    adc_range: calibrated. Returns (calibrated layer names, seconds). MAC histograms are cached in
    cache_dir (default: cimsim.calib_cache.default_cache_dir(); None = no cache). info, if given,
    receives the cache report (hit / partial / miss / off)."""
    if not any(getattr(m, "cfg", None) is not None and needs_calibration(m.cfg) for m in model.modules()):
        return [], 0.0
    tc = time.time()
    train_ids = load_wikitext2_ids(tok, split="train")[:calib_tokens]
    cal_len = min(seq_len, calib_tokens)

    def run_calib():
        with torch.no_grad():
            for i in range(0, train_ids.numel() - cal_len + 1, cal_len):
                model(train_ids[i:i + cal_len].unsqueeze(0).to(device))

    from cimsim.calib_cache import default_cache_dir, tensor_key
    if cache_dir is _DEFAULT:
        cache_dir = default_cache_dir()
    data_key = f"wikitext2-train:{tensor_key(train_ids)}:{cal_len}"
    report = {}
    print(f"calibrating (ADC range / statistical error model) on {calib_tokens} train tokens ...", flush=True)
    calibrated = calibrate_adc_range(model, run_calib, cache_dir=cache_dir, data_key=data_key, report=report)
    secs = round(time.time() - tc, 1)
    print(f"calibrated {len(calibrated)} layers in {secs} s (histogram cache: {report.get('cache')}, "
          f"{report.get('hits', 0)} layers from cache)", flush=True)
    if info is not None:
        info.update(report)
    return calibrated, secs
