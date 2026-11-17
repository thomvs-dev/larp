import sys, torch, dataclasses
sys.path.insert(0, '/home/claude/larp')
from eval.runner import load_model, swap_from_config, calibrate_if_needed
from eval.perplexity import load_wikitext2_ids
from cimsim.events import EventCounter
from cimsim.core.layers import _CIMBase
from cimsim.core.executor_fast import cim_matmul_int_fast
torch.set_num_threads(2)
variant = sys.argv[1]; seed = int(sys.argv[2])
model, tok, _ = load_model('/home/claude/hf_local/opt-125m', 'cpu'); model.eval()
swap_from_config(model, f'/home/claude/scratch/l1_st_s{seed}.yaml', EventCounter())
calibrate_if_needed(model, tok, 2048, 2048, 'cpu')
ids = load_wikitext2_ids(tok)
orig = _CIMBase._int_matmul
gen = torch.Generator().manual_seed(500 + seed)
diff = {}
def patched(self, qx, qw):
    if self._calibrating or 'layers.1.fc1' not in self.name:
        return orig(self, qx, qw)
    if variant in ('ba', 'baL'):
        g = gen if variant == 'ba' else self._generator(qx.device)
        return cim_matmul_int_fast(qx, qw, self.cfg, g, full_scale=self.adc_fs, chunks=self._chunks()).float()
    if variant == 'st':
        return orig(self, qx, qw)
    cfg0 = dataclasses.replace(self.cfg, rand_noise_sigma=0.0)
    base_ba = cim_matmul_int_fast(qx, qw, cfg0, None, full_scale=self.adc_fs, chunks=self._chunks()).double()
    ys = orig(self, qx, qw).double()                       # stat noisy
    c = self.cfg; self.cfg = cfg0; self._sat_plan = None
    base_st = orig(self, qx, qw).double()                  # stat noise-free
    self.cfg = c; self._sat_plan = None
    flips_st = ys - base_st
    yb = cim_matmul_int_fast(qx, qw, self.cfg, gen, full_scale=self.adc_fs, chunks=self._chunks()).double()
    flips_ba = yb - base_ba
    diff['base'] = float((base_st - base_ba).abs().max()); diff['n_base'] = int(((base_st - base_ba).abs() > 0.5).sum())
    out = {'st': base_st + flips_st, 'ba': base_ba + flips_ba, 'stbase_baflips': base_st + flips_ba,
           'babase_stflips': base_ba + flips_st}[variant]
    return out.float()
_CIMBase._int_matmul = patched
nll = []
with torch.no_grad():
    for w in range(2):
        x = ids[w * 2048:(w + 1) * 2048].unsqueeze(0)
        nll.append(float(model(x, labels=x).loss))
print(variant, seed, 'ppl', round(float(torch.tensor(nll).mean().exp()), 2), diff)
