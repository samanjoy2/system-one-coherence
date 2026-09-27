from __future__ import annotations

import os
import time
import logging
from pathlib import Path

import numpy as np
import requests

from .core import ROOT, digest, read_json, save_json, questions

os.environ.setdefault("HF_HOME", str(ROOT/".cache/huggingface"))
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TRITON_CACHE_DIR",str(ROOT/".cache/triton"))


class _CastWarningFilter(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("MatMul8bitLt: inputs will be cast")


# bitsandbytes logs this expected cast on every projection of every forward pass.
# The int8 backend still performs the cast; silence only the duplicate diagnostic.
logging.getLogger("bitsandbytes.autograd._functions").addFilter(_CastWarningFilter())


def payload(task):
    return {"type":"choice", "instructions":task["instructions"], "criteria":task["criteria"]}


class Jev:
    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv(ROOT/".env")
        self.session = requests.Session()
        self.session.headers.update({"Authorization":"Bearer " + os.environ["JEV_API_KEY"]})
        self.base = "https://api.typesafe.ai/v1"
        self.model = "jev-1.13.0"
        self.meta = {"model":self.model, "backend":"typesafe-http", "precision":"provider"}

    def predict(self, task):
        request = {"model":self.model, "state":task["state"], "questions":{"classification":payload(task)}}
        for attempt in range(5):
            try:
                r = self.session.post(self.base+"/systemone",json=request,timeout=90)
            except (requests.Timeout, requests.ConnectionError):
                if attempt == 4: raise
                time.sleep(min(2**attempt,16))
                continue
            if r.status_code in (429,500,502,503,504):
                if attempt == 4: r.raise_for_status()
                time.sleep(min(float(r.headers.get("Retry-After",2**attempt)),30))
                continue
            r.raise_for_status()
            data = r.json()
            if data.get("model") not in (None,self.model):
                raise ValueError(f"Jev revision changed: {data.get('model')}")
            return data["answers"]["classification"]["probabilities"], {"request":request,"response":data}
        raise RuntimeError("No Jev response")


class Qwen:
    def __init__(self, precision="int8"):
        import torch
        from .attention import configure
        attention_meta = configure(precision)
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        lock = read_json(ROOT/"sources.lock.json")
        repo = "Qwen/Qwen3.5-4B-Base"
        rev = lock["models"][repo]
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(repo,revision=rev)
        kw = {"revision":rev,"dtype":torch.bfloat16 if precision=="int8" else torch.float32,
              "attn_implementation":"sdpa", "device_map":"auto", "max_memory":{0:"6GiB","cpu":"9GiB"}}
        if precision == "int8": kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(repo,**kw).eval()
        self.meta = {"model":repo,"revision":rev,"precision":precision,"backend":"transformers",
                     "scoring":"sum conditional candidate-ID log likelihood; no EOS; no length correction"}
        self.meta.update(attention_meta)
        self.meta['likelihood_execution'] = 'cached autoregressive; independent sequential validation per candidate count'
        self.meta['trie_branch_batch_size'] = 1
        self.cache_validated = set()

    def predict(self, task):
        torch = self.torch
        prompt = ("Input:\n" + task["state"] + "\n\n" + task["instructions"] + "\nOptions:\n" +
                  "\n".join(f"{k}: {v}" for k,v in task["criteria"].items()) + "\nAnswer:")
        prefix = self.tokenizer.encode(prompt,add_special_tokens=False)
        # Explicit token concatenation defines the boundary without tokenizer merges.
        candidates = [self.tokenizer.encode(" " + k,add_special_tokens=False) for k in task["criteria"]]
        scores = []
        device = self.model.get_input_embeddings().weight.device
        with torch.inference_mode():
            import copy
            out = self.model(input_ids=torch.tensor([prefix],device=device),use_cache=True,logits_to_keep=1)
            scores = [0.] * len(candidates)
            def visit(groups, depth, output, accumulated):
                logp = torch.log_softmax(output.logits[:,-1].float(),dim=-1)
                pending = []
                for row,indices in enumerate(groups):
                    branches = {}
                    for i in indices: branches.setdefault(candidates[i][depth],[]).append(i)
                    for token, group in branches.items():
                        value = accumulated[row] + logp[row,token].item()
                        remaining = []
                        for i in group:
                            if len(candidates[i]) == depth+1: scores[i] = value
                            else: remaining.append(i)
                        if remaining: pending.append((row,token,remaining,value))
                # Int8 batched rows change activation quantization; keep branches single-row.
                for start in range(0,len(pending)):
                    batch = pending[start:start+1]
                    cache = copy.deepcopy(output.past_key_values)
                    cache.reorder_cache(torch.tensor([b[0] for b in batch],device=device))
                    nxt = self.model(input_ids=torch.tensor([[b[1]] for b in batch],device=device),
                                     past_key_values=cache,use_cache=True,logits_to_keep=1)
                    visit([b[2] for b in batch],depth+1,nxt,[b[3] for b in batch])
            visit([list(range(len(candidates)))],0,out,[0.])
            cache_check = None
            if len(candidates) not in self.cache_validated:
                deviations = []
                sequential_deviations = []
                for i in sorted({0,len(candidates)//2,len(candidates)-1}):
                    tail = candidates[i]
                    reference = self.model(input_ids=torch.tensor([prefix+tail],device=device),
                                           use_cache=False,logits_to_keep=len(tail)+1)
                    z = torch.log_softmax(reference.logits[0,:-1].float(),dim=-1)
                    expected = z.gather(1,torch.tensor(tail,device=z.device)[:,None]).sum().item()
                    deviations.append(abs(expected-scores[i]))
                    step = self.model(input_ids=torch.tensor([prefix],device=device),use_cache=True,logits_to_keep=1)
                    sequential = 0.
                    for j,token in enumerate(tail):
                        sequential += torch.log_softmax(step.logits[0,-1].float(),dim=-1)[token].item()
                        if j+1<len(tail):
                            step = self.model(input_ids=torch.tensor([[token]],device=device),
                                              past_key_values=step.past_key_values,use_cache=True,logits_to_keep=1)
                    sequential_deviations.append(abs(sequential-scores[i]))
                cache_check = {'uncached_max':max(deviations),'independent_sequential_max':max(sequential_deviations)}
                save_json(ROOT/'results/qwen_cache_checks'/digest(self.meta)[:16]/(str(len(candidates))+'.json'),{'backend':self.meta,
                          'uncached_errors':deviations,'independent_sequential_errors':sequential_deviations})
                if max(sequential_deviations) > .05:
                    raise ValueError(f"Trie versus independent sequential log-likelihood differs by {max(sequential_deviations)}")
                self.cache_validated.add(len(candidates))
        scores = np.array(scores,dtype=float)
        exp = np.exp(scores-scores.max())
        p = exp/exp.sum()
        lengths = np.array([len(t) for t in candidates])
        normalized = scores/lengths
        alt = np.exp(normalized-normalized.max()); alt /= alt.sum()
        return dict(zip(task["criteria"],p.tolist())), {"prompt":prompt,"prefix_tokens":len(prefix),
                   "candidate_tokens":candidates,"log_likelihoods":scores.tolist(),
                   "cache_validation_max_loglikelihood_error":cache_check,
                   "length_normalized_probabilities":dict(zip(task["criteria"],alt.tolist()))}


class Laya:
    def __init__(self):
        import torch
        import laya
        from huggingface_hub import snapshot_download
        repo = "convaiinnovations/laya"
        rev = read_json(ROOT/"sources.lock.json")["models"][repo]
        path = snapshot_download(repo,revision=rev,ignore_patterns=["multilingual/*","typed-decisions/*","*.onnx"])
        self.agent = laya.load(path,device="cuda")
        self.agent.amp_enabled = False
        self.agent.dtype = torch.float32
        self.agent.model.float().eval()
        self.meta = {"model":repo,"revision":rev,"precision":"fp32","backend":"laya",
                     "temperature":self.agent.temperature,"temperature_by_options":self.agent.temperature_by_options}
        self.preflight()

    def preflight(self):
        from laya.common import render_options, build_sequence
        tok = self.agent.tok
        encode = lambda s: tok(s,add_special_tokens=False)["input_ids"]
        tax = read_json(ROOT/"data/taxonomies.json")
        rows = read_json(ROOT/"data/examples.json")
        maximum_head = maximum_state = 0
        max_option = 0
        # Length is independent of option permutation for this tokenizer/formatter.
        for dataset, taxonomy in tax.items():
            sample = next(e for e in rows if e["dataset"] == dataset)
            for task in questions(sample,taxonomy):
                q = self.agent._to_internal(payload(task))
                opts = render_options(q)
                lengths = [len(encode(" "+o)) for o in opts]
                max_option = max(max_option,max(lengths))
                maximum_head = max(maximum_head,len(encode("choice question: "+q["ins"]))+sum(n+1 for n in lengths)+16)
        for row in rows:
            maximum_state = max(maximum_state,len(encode(row["text"])))
        if max_option > 48:
            raise ValueError(f"Laya truncates an option at 48 tokens: required {max_option}")
        head = ((maximum_head+63)//64)*64
        context = ((head+maximum_state+4+63)//64)*64
        native_limit = self.agent.model.encoder.config.max_position_embeddings
        if context > native_limit:
            raise ValueError(f"Laya needs {context} tokens, architecture supports {native_limit}")
        self.agent.cfg.update(head_max_len=head,max_len=context)
        self.meta.update(head_max_len=head,max_len=context,encoder_max_length=native_limit)
        # Verify exact complete token sequences at inference too, including all permutations.
        save_json(ROOT/"results/laya_preflight.json",self.meta | {"max_option_tokens":max_option})

    def predict(self,task):
        from laya.common import render_options, build_sequence
        tok = self.agent.tok
        q = self.agent._to_internal(payload(task))
        encode = lambda s: tok(s,add_special_tokens=False)["input_ids"]
        expected = [tok.cls_token_id]+encode("choice question: "+q["ins"])+[tok.sep_token_id]
        for option in render_options(q): expected += [tok.mask_token_id]+encode(" "+option)
        expected += [tok.sep_token_id]+encode(task["state"])+[tok.sep_token_id]
        actual, markers = build_sequence(tok,task["state"],q,self.agent.cfg["max_len"],self.agent.cfg["head_max_len"])
        if actual != expected or len(markers) != len(task["criteria"]):
            raise ValueError("Laya tokenization truncates or rewrites prompt content")
        result = self.agent.predict(task["state"],{"classification":payload(task)})
        if str(self.agent.device) != "cuda":
            raise RuntimeError("Laya changed device during inference")
        return result["answers"]["classification"]["probabilities"], {"response":result,"encoded_tokens":len(actual)}


def backend(name,precision="int8"):
    if name not in ("jev", "laya"):
        raise ValueError("This release supports Jev and Laya only")
    if name == "jev": return Jev()
    if name == "qwen": return Qwen(precision)
    if name == "laya": return Laya()
    if name == "kev":
        from .kev_backend import Kev
        return Kev(precision)
    raise ValueError(name)
