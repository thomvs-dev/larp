"""A3 real-model check of statistical mode v2 (OPT-125M, f5 config: calibrated ADC 6 b, no noise):
per-layer statistical vs bit-accurate on one window, margin filter vs strict saturation bound,
statistical PPL on the first 8,192 test tokens. Writes <out_dir>/check.json.
Usage: CIMSIM_WIKITEXT_DIR=<dir with train/test.parquet> python scripts/a3_real_check.py <model_dir> <out_dir>"""
import json, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from cimsim.core.layers import _CIMBase
from cimsim.core.executor_fast import cim_matmul_int_fast
from cimsim.stat.saturation import saturation_correction
from cimsim.events import EventCounter
from eval.perplexity import load_wikitext2_ids, perplexity
from eval.runner import calibrate_if_needed, load_model, swap_from_config
import dataclasses

model_dir, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
model, tok, _ = load_model(model_dir, "cpu"); model.eval()
swap_from_config(model, str(ROOT / "config/experiments/f5_adc6_calibrated.yaml"), EventCounter())
cim = [m for m in model.modules() if isinstance(m, _CIMBase)]
for m in cim:
    m.cfg = dataclasses.replace(m.cfg, mode="statistical"); m._gen = None
t = time.time()
_, cs = calibrate_if_needed(model, tok, 2048, 2048, "cpu")
log = {"calib_seconds": cs}
log["stat_params"] = {m.name: m.stat_info for m in cim}
print("calibrated", cs, flush=True)
ids = load_wikitext2_ids(tok)

# 1) per-layer check on the first window (2,048 tokens): stat v2 vs bit-accurate, margin vs strict
per_layer = {}
orig = _CIMBase._int_matmul
def patched(self, qx, qw):
    y = orig(self, qx, qw)
    if self._calibrating or self.cfg.mode != "statistical":
        return y
    fs = self.adc_fs
    ba = cim_matmul_int_fast(qx, qw, self.cfg, self._generator(qx.device), full_scale=fs, chunks=self._chunks())
    strict = saturation_correction(qx, qw, self.cfg, self._saturation_tables(), self._chunks())
    marg = saturation_correction(qx, qw, self.cfg, self._saturation_tables(), self._chunks(), col_mask=self.sat_cols)
    ideal = qx.double() @ qw.double().t()
    e_ba = ba - ideal
    z = torch.zeros_like(ideal)
    s_strict = z if strict is None else strict
    s_marg = z if marg is None else marg
    per_layer[self.name] = dict(
        ba_err_rms=float(e_ba.pow(2).mean().sqrt()), ideal_rms=float(ideal.pow(2).mean().sqrt()),
        stat_vs_ba_rms=float((y.double() - ba).pow(2).mean().sqrt()),
        strict_vs_ba_rms=float((ideal + s_strict - ba).pow(2).mean().sqrt()),
        missed_outputs=int(((s_strict - s_marg).abs() > 0).sum()),
        missed_rms=float((s_strict - s_marg).pow(2).mean().sqrt()),
        ba_outputs_with_error=int((e_ba.abs() > 1e-6).sum()), outputs=ideal.numel(),
        sat_cols_frac=(self.stat_info or {}).get("sat_cols_frac"))
    return y
_CIMBase._int_matmul = patched
t = time.time()
with torch.no_grad():
    model(ids[:2048].unsqueeze(0))
log["per_layer_seconds"] = round(time.time() - t, 1)
log["per_layer"] = per_layer
_CIMBase._int_matmul = orig
(out / "check.json").write_text(json.dumps(log, indent=1))
print("per-layer check done", log["per_layer_seconds"], flush=True)

# 2) statistical v2 PPL, first 8,192 test tokens, seeds 0-2 (noise-free config: seeds should agree)
res = {}
for s in (0,):
    for m in cim:
        m.cfg = dataclasses.replace(m.cfg, seed=s); m._gen = None; m.sat_stats = {}
    t = time.time()
    r = perplexity(model, ids, 2048, 4, "cpu", progress=False)
    dt = time.time() - t
    ex = sum(m.sat_stats.get("macs_exact", 0) for m in cim); tot = sum(m.sat_stats.get("macs_total", 0) for m in cim)
    res[f"seed{s}"] = dict(ppl=r["ppl"], seconds=round(dt, 1), ms_per_token=round(1000 * dt / 8192, 2),
                           sat_work_fraction=ex / tot)
    print(res[f"seed{s}"], flush=True)
    log["statistical_v2"] = res
    (out / "check.json").write_text(json.dumps(log, indent=1))
for m in cim:
    m.cfg = dataclasses.replace(m.cfg, mode="fake_quant")
t = time.time(); r = perplexity(model, ids, 2048, 4, "cpu", progress=False)
log["fake_quant"] = dict(ppl=r["ppl"], seconds=round(time.time() - t, 1))
(out / "check.json").write_text(json.dumps(log, indent=1))
print("done", log["fake_quant"], flush=True)
