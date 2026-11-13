"""A5: size of the statistical-mode saturation plan on Qwen2.5-1.5B-shaped layers (synthetic weights and
inputs) at calibrated ADC 6 b (lossless) and 5 b (every column saturates). Prints index memory per layer.

    python scripts/a5_plan_mem.py > results/a5_plan_mem_sandbox/result.txt
"""
import sys, torch, time
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
import cimsim.core.layers as L
from cimsim.config import CIMConfig
from cimsim.core.layers import CIMLinear
orig=L.full_scale_from_histogram
def run(K,N,bits,fs):
    L.full_scale_from_histogram=lambda h,cfg: torch.full_like(orig(h,cfg), float(fs))
    torch.manual_seed(0); lin=torch.nn.Linear(K,N)
    st=CIMLinear.from_linear(lin, CIMConfig(mode="statistical", w_bits=8,x_bits=8,nrow=128,signed_act=True,w_gran="per_channel",x_gran="per_token",adc_bits=bits,adc_range="calibrated"), name="L")
    g=torch.Generator().manual_seed(1); x=torch.randn(256,K,generator=g); x[:, :4]*=25
    st.start_calibration(); st(x); st.finish_calibration()
    t=time.time(); st(x[:64]); dt=time.time()-t
    plan=st._sat_plan[1] if st._sat_plan else None
    if plan is None: print(K,N,bits,"no plan"); return
    idx=sum(e[3].numel()+e[4].numel() for e in plan["entries"])*8
    print(f"K {K} N {N} adc {bits}: entries {len(plan['entries'])} index {idx/1e6:.0f} MB, bit cache {plan['cache_bytes']/1e6:.0f} MB, cached {plan['cached']}, sat_cols {st.stat_info.get('sat_cols_frac'):.3f}, 64 tok {dt:.1f}s")
run(1536,8960,6,63); run(1536,8960,5,31); run(8960,1536,5,31); run(1536,1536,5,31)
