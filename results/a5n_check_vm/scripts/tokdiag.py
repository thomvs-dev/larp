import sys, json, torch
sys.path.insert(0, '/home/claude/larp')
from eval.runner import load_model, swap_from_config, calibrate_if_needed
from eval.perplexity import load_wikitext2_ids
from cimsim.events import EventCounter
from cimsim.core.layers import _CIMBase
torch.set_num_threads(2)
cfg = sys.argv[1]
model, tok, _ = load_model('/home/claude/hf_local/opt-125m', 'cpu'); model.eval()
swap_from_config(model, cfg, EventCounter())
calibrate_if_needed(model, tok, 2048, 2048, 'cpu')
ids = load_wikitext2_ids(tok)[:2048].unsqueeze(0)
rec = {}
orig = _CIMBase._int_matmul
def patched(self, qx, qw):
    y = orig(self, qx, qw)
    if not self._calibrating and 'layers.1.fc1' in self.name:
        ideal = qx.double() @ qw.double().t()
        e = y.double() - ideal
        rec['e'] = e; rec['ideal'] = ideal
    return y
_CIMBase._int_matmul = patched
with torch.no_grad():
    out = model(ids)
lp = torch.log_softmax(out.logits[0, :-1].double(), -1)
nll = -lp.gather(1, ids[0, 1:].unsqueeze(1)).squeeze(1)
torch.save({'nll': nll, 'e': rec['e'].float(), 'ideal': rec['ideal'].float()}, sys.argv[2])
print('ppl', float(nll.mean().exp()))
