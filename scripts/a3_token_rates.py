# Per-position flip rates at rn010 on real OPT-125M activations: bit-accurate vs statistical v3 (thinning)
# vs uniform placement (v3 before thinning, emulated by pmax-only acceptance). Position 0 = first token of the window.
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
model, tok, _ = load_model("/home/claude/hf_local/opt-125m", "cpu"); model.eval()
swap_from_config(model, str(ROOT / "config/experiments/a3g_adc6cal_all_rn010.yaml"), EventCounter())
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
    nb = (ba - ideal).abs() > 1e-6
    ns = (y.double() - ideal).abs() > 1e-6
    big_b = (ba - ideal).abs() >= 4096; big_s = (y.double() - ideal).abs() >= 4096
    out[self.name] = dict(ba_pos0=float(nb[0].double().mean()), ba_rest=float(nb[1:].double().mean()),
                          st_pos0=float(ns[0].double().mean()), st_rest=float(ns[1:].double().mean()),
                          ba_big_pos0=int(big_b[0].sum()), st_big_pos0=int(big_s[0].sum()),
                          ba_big_rest=float(big_b[1:].double().mean()), st_big_rest=float(big_s[1:].double().mean()))
    return y
_CIMBase._int_matmul = patched
with torch.no_grad(): model(ids[:512].unsqueeze(0))
Path("/home/claude/scratch/token_rates.json").write_text(json.dumps(out, indent=1))
print("done")
