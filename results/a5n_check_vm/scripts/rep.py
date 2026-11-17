import sys, torch, time
sys.path.insert(0, '/home/claude/larp')
from eval.runner import load_model, swap_from_config, calibrate_if_needed
from eval.perplexity import load_wikitext2_ids
from cimsim.events import EventCounter
from cimsim.core.layers import _CIMBase
from cimsim.core.executor_fast import cim_matmul_int_fast
torch.set_num_threads(2)
model, tok, _ = load_model('/home/claude/hf_local/opt-125m', 'cpu'); model.eval()
swap_from_config(model, '/home/claude/scratch/l1_st_s0.yaml', EventCounter())
calibrate_if_needed(model, tok, 2048, 2048, 'cpu')
ids = load_wikitext2_ids(tok)[:2048].unsqueeze(0)
cap = {}
orig = _CIMBase._int_matmul
def patched(self, qx, qw):
    if not self._calibrating and 'layers.1.fc1' in self.name and 'qx' not in cap:
        cap['qx'] = qx.clone(); cap['qw'] = qw.clone(); cap['layer'] = self
    return orig(self, qx, qw)
_CIMBase._int_matmul = patched
with torch.no_grad(): model(ids[:, :256])
L = cap['layer']; qx = cap['qx'][:96]; qw = cap['qw']
ideal = qx.double() @ qw.double().t()
R = int(sys.argv[1])
sb = torch.zeros_like(ideal); sb2 = torch.zeros_like(ideal); ss = torch.zeros_like(ideal); ss2 = torch.zeros_like(ideal)
L._gen = None
t0 = time.time()
for r in range(R):
    yb = cim_matmul_int_fast(qx, qw, L.cfg, torch.Generator().manual_seed(1000 + r), full_scale=L.adc_fs, chunks=L._chunks())
    ys = orig(L, qx, qw).double()
    eb = yb - ideal; es = ys - ideal
    sb += eb; sb2 += eb * eb; ss += es; ss2 += es * es
print('time', time.time() - t0)
mb, ms = sb / R, ss / R
vb, vs = sb2 / R - mb ** 2, ss2 / R - ms ** 2
print('mean bias BA', float(mb.mean()), 'ST', float(ms.mean()))
print('var total BA', float(vb.mean()), 'ST', float(vs.mean()))
print('corr of bias maps', float(torch.corrcoef(torch.stack([mb.flatten(), ms.flatten()]))[0, 1]))
print('corr of var maps', float(torch.corrcoef(torch.stack([vb.flatten(), vs.flatten()]))[0, 1]))
# by token
print('per-token var ratio ST/BA (first 12):', [round(float(vs[t].mean() / vb[t].mean()), 2) for t in range(12)])
d = (vs.mean(1) / vb.mean(1))
print('per-token var ratio min/max', float(d.min()), float(d.max()))
torch.save({'mb': mb, 'ms': ms, 'vb': vb, 'vs': vs, 'qx': qx, 'ideal': ideal}, '/home/claude/scratch/rep.pt')
