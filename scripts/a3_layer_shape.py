# Usage: CIMSIM_WIKITEXT_DIR=<dir> python scripts/a3_layer_shape.py <config name without .yaml> <tokens>  (paths below assume the cloud sandbox)
# Layer-level shape of the analog-noise error at rn010 on real OPT-125M activations (one window, 2,048 test tokens):
# bit-accurate noisy error vs the statistical model's Gaussian. Stats per layer: std, excess kurtosis,
# fraction of outputs with any error.
import dataclasses, json, sys, time
from pathlib import Path
import torch
ROOT = Path("/home/claude/larp"); sys.path.insert(0, str(ROOT))
from cimsim.core.layers import _CIMBase
from cimsim.core.executor_fast import cim_matmul_int_fast
from cimsim.events import EventCounter
from eval.perplexity import load_wikitext2_ids
from eval.runner import calibrate_if_needed, load_model, swap_from_config
torch.set_num_threads(2)
cfgname = sys.argv[1]; ntok = int(sys.argv[2])
model, tok, _ = load_model("/home/claude/hf_local/opt-125m", "cpu"); model.eval()
swap_from_config(model, str(ROOT / f"config/experiments/{cfgname}.yaml"), EventCounter())
cim = [m for m in model.modules() if isinstance(m, _CIMBase)]
for m in cim: m.cfg = dataclasses.replace(m.cfg, mode="statistical"); m._gen = None
calibrate_if_needed(model, tok, 2048, 2048, "cpu")
ids = load_wikitext2_ids(tok)
out = {}
orig = _CIMBase._int_matmul
def patched(self, qx, qw):
    y = orig(self, qx, qw)
    if self._calibrating: return y
    ideal = qx.double() @ qw.double().t()
    ba = cim_matmul_int_fast(qx, qw, self.cfg, torch.Generator().manual_seed(1), full_scale=self.adc_fs, chunks=self._chunks())
    e = ba - ideal; es = y.double() - ideal
    def st(v):
        v = v.flatten(); c = v - v.mean(); var = float((c * c).mean())
        return dict(std=var ** 0.5, kurt=float((c ** 4).mean() / var ** 2 - 3) if var > 0 else 0.0,
                    frac_nonzero=float((v.abs() > 1e-6).double().mean()), max_abs=float(v.abs().max()))
    out[self.name] = dict(ba=st(e), stat=st(es), ideal_rms=float(ideal.pow(2).mean().sqrt()))
    return y
_CIMBase._int_matmul = patched
t = time.time()
with torch.no_grad(): model(ids[:ntok].unsqueeze(0))
Path(f"/home/claude/scratch/{cfgname}_layers.json").write_text(json.dumps(out, indent=1))
print("done", round(time.time() - t, 1))
