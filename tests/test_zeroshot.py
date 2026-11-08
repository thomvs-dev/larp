"""Zero-shot wrapper (eval/zeroshot.py) end to end on a tiny model and a local lm-eval task (no downloads)."""

import json
import sys
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
pytest.importorskip("lm_eval")
tokenizers = pytest.importorskip("tokenizers")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # eval/ is not an installed package

from cimsim import calibrate_adc_range  # noqa: E402
from cimsim.config import LayerConfigMap  # noqa: E402
from cimsim.events import EventCounter  # noqa: E402
from cimsim.models.swap import swap_linears  # noqa: E402
from eval.zeroshot import evaluate_tasks, mcnemar_p, paired_compare  # noqa: E402
from test_swap import tiny  # noqa: E402

CHARS = list("abcdefghijklmnopqrstuvwxyz ?.:\n")


def char_tokenizer():
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    vocab = {"<unk>": 0, "<eos>": 1, **{c: i + 2 for i, c in enumerate(CHARS)}}
    t = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")
    return transformers.PreTrainedTokenizerFast(tokenizer_object=t, unk_token="<unk>", eos_token="<eos>",
                                                pad_token="<eos>")


@pytest.fixture(scope="module")
def task_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("zs_task")
    g = torch.Generator().manual_seed(0)
    with open(d / "data.jsonl", "w") as f:
        for i in range(12):
            words = ["".join(CHARS[int(j)] for j in torch.randint(0, 26, (4,), generator=g)) for _ in range(3)]
            f.write(json.dumps({"question": f"question {i}: which word?", "choices": words, "label": i % 3}) + "\n")
    (d / "tiny_mc.yaml").write_text(
        "task: tiny_mc\n"
        "dataset_path: json\n"
        "dataset_kwargs:\n"
        f"  data_files:\n    test: {(d / 'data.jsonl').as_posix()}\n"
        "test_split: test\n"
        "output_type: multiple_choice\n"
        "doc_to_text: \"{{question}}\\nanswer:\"\n"
        "doc_to_choice: \"{{choices}}\"\n"
        "doc_to_target: \"{{label}}\"\n"
        "target_delimiter: \" \"\n"
        "metric_list:\n  - metric: acc\n  - metric: acc_norm\n")
    return d


def run(model, out, task_dir, counter=None, limit=None, batch_size=1):
    return evaluate_tasks(model, char_tokenizer(), ["tiny_mc"], out, counter=counter, limit=limit,
                          include_path=str(task_dir), resume=False, batch_size=batch_size)["tiny_mc"]


def test_fp_swap_matches_unswapped_model(tmp_path, task_dir):
    ref = run(tiny("llama").eval(), tmp_path / "ref", task_dir)
    model = tiny("llama").eval()
    swap_linears(model, LayerConfigMap.from_dict({"default": {"mode": "fp"}}))
    got = run(model, tmp_path / "fp", task_dir)
    assert ref["n"] == got["n"] == 12
    assert ref["acc"] == got["acc"] and ref["acc_norm"] == got["acc_norm"]
    assert got["model_tokens"] > 0 and got["forward_calls"] == 36          # 12 docs x 3 choices, batch 1


def test_statistical_with_calibration_and_attention(tmp_path, task_dir):
    from cimsim.core.attention import swap_attention
    model = tiny("qwen2").eval()
    counter = EventCounter()
    m = LayerConfigMap.from_dict({"default": {"mode": "statistical", "nrow": 32, "adc_bits": 6,
                                              "adc_range": "calibrated", "rand_noise_sigma": 0.05}})
    swap_linears(model, m, counter)
    swap_attention(model, m, counter)
    ids = torch.randint(2, 33, (1, 48))
    assert len(calibrate_adc_range(model, lambda: model(ids))) > 0
    s = run(model, tmp_path / "stat", task_dir, counter=counter, limit=8)
    assert s["n"] == 8 and 0.0 <= s["acc"] <= 1.0
    ev = json.loads((tmp_path / "stat" / "events_tiny_mc.json").read_text())
    assert ev["macs"] > 0 and ev["adc_conversions"] > 0
    rec = json.loads((tmp_path / "stat" / "zs_tiny_mc.json").read_text())
    assert [d["doc_id"] for d in rec["docs"]] == list(range(8))


def test_batched_noise_free_statistical_equals_batch_1(tmp_path, task_dir):
    """Linear layers only: right padding in a batch does not change the scores (noise-free)."""
    model = tiny("llama").eval()
    swap_linears(model, LayerConfigMap.from_dict({"default": {"mode": "statistical", "nrow": 32, "adc_bits": 6,
                                                              "adc_range": "calibrated"}}))
    ids = torch.randint(2, 33, (1, 48))
    calibrate_adc_range(model, lambda: model(ids))
    a = json.loads(json.dumps(run(model, tmp_path / "b1", task_dir)))
    run(model, tmp_path / "b4", task_dir, batch_size=4)
    da = json.loads((tmp_path / "b1" / "zs_tiny_mc.json").read_text())["docs"]
    db = json.loads((tmp_path / "b4" / "zs_tiny_mc.json").read_text())["docs"]
    assert da == db and a["n"] == 12


def test_batch_rejected_with_cim_attention(tmp_path, task_dir):
    from cimsim.core.attention import swap_attention
    model = tiny("llama").eval()
    m = LayerConfigMap.from_dict({"default": {"mode": "fake_quant"}})
    swap_linears(model, m)
    swap_attention(model, m)
    with pytest.raises(ValueError):
        run(model, tmp_path / "x", task_dir, batch_size=4)


def test_resume_skips_finished_task(tmp_path, task_dir):
    model = tiny("llama").eval()
    tok = char_tokenizer()
    first = evaluate_tasks(model, tok, ["tiny_mc"], tmp_path, limit=4, include_path=str(task_dir))
    again = evaluate_tasks(model, tok, ["tiny_mc"], tmp_path, limit=4, include_path=str(task_dir))
    assert again["tiny_mc"]["seconds"] == first["tiny_mc"]["seconds"]      # read back, not rerun


def bos_char_tokenizer():
    """Character tokenizer whose encode() prepends <s> (id 34), like the Llama tokenizers in transformers 5."""
    from tokenizers import processors
    tok = char_tokenizer()
    tok.add_special_tokens({"bos_token": "<s>"})
    tok.backend_tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A", special_tokens=[("<s>", tok.bos_token_id)])
    assert tok.encode("ab")[0] == tok.bos_token_id
    return tok


@pytest.mark.parametrize("add_bos", [False, True])
def test_bos_setting_reaches_model_inputs(tmp_path, task_dir, add_bos):
    """Default: no BOS in any request even if the tokenizer adds one (diag_zs_tokens_tinyllama);
    add_bos_token=True: every request starts with BOS."""
    model = tiny("llama").eval()
    tok = bos_char_tokenizer()
    firsts = []
    h = model.register_forward_pre_hook(lambda m, a, k: firsts.extend((a[0] if a else k["input_ids"])[:, 0].tolist()),
                                        with_kwargs=True)
    kw = {"add_bos_token": True} if add_bos else {}
    s = evaluate_tasks(model, tok, ["tiny_mc"], tmp_path, limit=4, include_path=str(task_dir), resume=False, **kw)
    h.remove()
    assert s["tiny_mc"]["n"] == 4 and len(firsts) == 12
    assert all((f == tok.bos_token_id) == add_bos for f in firsts)
    rec = json.loads((tmp_path / "zs_tiny_mc.json").read_text())
    assert rec["add_bos_token"] is add_bos


def test_paired_compare_and_mcnemar():
    a = [{"doc_id": i, "acc": int(i % 2 == 0)} for i in range(10)]
    b = [{"doc_id": i, "acc": int(i % 2 == 0 or i == 1)} for i in range(10)]
    c = paired_compare(a, a)
    assert c["diff"] == 0 and c["mcnemar_p"] == 1.0
    c = paired_compare(a, b)
    assert c["b_only"] == 1 and c["a_only"] == 0 and abs(c["diff"] - 0.1) < 1e-12
    assert mcnemar_p(0, 6) == pytest.approx(2 / 64)
