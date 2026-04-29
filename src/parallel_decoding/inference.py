import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["CUDA_VISIBLE_DEVICES"] = "5,1,2,3"

# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,3,5,6"

import torch
import argparse
import contexttimer
import gc
from colorama import Fore, Style
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from  tqdm import tqdm
from utils import (
    read_json,
    save_json
)
from peft import PeftModel, LoraConfig, get_peft_model
import re
from typing import Any, Dict, List, Optional, Sequence

torch.manual_seed(520)
from argparse import ArgumentParser
from pathlib import Path
from speculative_sampling import (
    autoregressive_sampling,
    efficient_edit_speculative_sampling,
    efficient_generation_speculative_sampling,
)
def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--inference_type', type=str)
    parser.add_argument('--output_file', type=Path)
    parser.add_argument('--data_file', type=Path)
    parser.add_argument('--draft_lora_path', type=Path, default=None)
    parser.add_argument('--draft_model', type=Path)
    parser.add_argument('--target_model', type=Path)
    parser.add_argument(
        '--save_prompt_logits',
        action='store_true',
        help='keep full prompt logits in KVCacheModel history; default keeps only the last prompt position to save memory',
    )
    return parser.parse_args()
import re
import tiktoken
FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)?\n(.*?)```", re.DOTALL)
# MAX_CROSS_FILE_TOKENS = 15000
MAX_CROSS_FILE_TOKENS = 10000

_CROSS_FILE_SEP = "#-------------------------------------------\n"
_FALLBACK_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|\S")
_ENCODER = tiktoken.get_encoding("cl100k_base") if tiktoken is not None else None
def _count_text_tokens(text: str) -> int:
    if not text:
        return 0
    if _ENCODER is not None:
        return len(_ENCODER.encode(text, disallowed_special=()))
    return len(_FALLBACK_TOKEN_RE.findall(text))


def _truncate_text_to_tokens(text: str, token_budget: int) -> str:
    if token_budget <= 0 or not text:
        return ""
    if _ENCODER is not None:
        token_ids = _ENCODER.encode(text, disallowed_special=())
        if len(token_ids) <= token_budget:
            return text
        return _ENCODER.decode(token_ids[:token_budget])
    tokens = _FALLBACK_TOKEN_RE.findall(text)
    if len(tokens) <= token_budget:
        return text
    return "".join(tokens[:token_budget])


def _truncate_blocks_to_budget(blocks: List[str], token_budget: int) -> List[str]:
    kept: List[str] = []
    used = 0
    for block in blocks:
        blk = block or ""
        block_tokens = _count_text_tokens(blk)
        if block_tokens == 0:
            continue
        if used + block_tokens <= token_budget:
            kept.append(blk)
            used += block_tokens
            continue
        remain = token_budget - used
        if remain > 0:
            kept.append(_truncate_text_to_tokens(blk, remain))
        break
    return kept


def _stable_bucket(token: str, dim: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim

import re
from typing import List

def extract_comments(code: str) -> List[str]:
    """
    提取 Python 代码中的注释（# 开头的行注释 + 三引号文档注释）。
    返回所有匹配到的注释内容。
    """

    # 匹配模式：
    # 1. 行注释：# 后面到行尾
    # 2. 多行注释 / docstring：''' ... ''' 或 """ ... """
    pattern = r"""
        \#.*?$                         |   
        (\"\"\"[\s\S]*?\"\"\")         |   
        (\'\'\'[\s\S]*?\'\'\')            
    """

    matches = re.findall(pattern, code, flags=re.MULTILINE | re.VERBOSE)

    comments = []
    for m in matches:
        # m 是一个 tuple，只有一个非空元素
        if isinstance(m, tuple):
            for part in m:
                if part:
                    comments.append(part)
        else:
            comments.append(m)

    return comments
def find_and_slice(a, b):
    """
    a: 目标字符串
    b: 字符串列表
    """
    for s in b:
        if a in s:
            # 找到 a 第一次出现的位置索引
            start_index = s.find(a)
            # 返回 a 之后的所有内容
            return s[start_index:]
    
    # 如果遍历完列表都没有匹配项，返回原始字符串 a
    return a
def code_generation_prompt(instruction):
    prompt = f"""###User:
You are a code completion assistant.
{instruction}.
Please output the complete function code directly.
###Assistant
```python
"""
    return prompt
def speculative_sampling_inference(target_model, draft_model, eos_token_id_tensor, input_ids , max_token = 4096, temperature= 0.2, top_p = 0.95,top_k = 5):
    with contexttimer.Timer() as t:
        with torch.no_grad():
            outputs = speculative_sampling_original(
                prefix = input_ids,
                approx_model = draft_model,
                target_model = target_model,
                eos_token_id_tensor = eos_token_id_tensor,
                max_len=1500,
                temperature = temperature, 
                top_k= top_k, 
                top_p = top_p)
    time = t.elapsed
    tokens = outputs.shape[-1] - input_ids.shape[-1]
    result = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
    return {"time":time, "tokens":tokens, "rate": tokens/time,"result":result}
def efficient_generation_inference(target_model, code_before, eos_token_id_tensor, input_ids , max_token = 4096, temperature= 0.2, top_p = 0.95,top_k = 5,
                                   save_prompt_logits: bool = False):
    precode = tokenizer.encode(code_before, add_special_tokens=False, return_tensors="pt").to(target_model.device)
    with contexttimer.Timer() as t:
        with torch.no_grad():
            outputs = efficient_generation_speculative_sampling(
                prefix = input_ids, 
                precode = precode, 
                target_model = target_model,
                eos_token_id_tensor = eos_token_id_tensor,
                max_len=1500 ,
                policy = "greedy",
                temperature = temperature, 
                top_k= top_k, 
                top_p = top_p)
    time = t.elapsed
    tokens = outputs.shape[-1] - input_ids.shape[-1]
    result = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
    return {"time":time, "tokens":tokens, "rate": tokens/time,"result":result}
def efficient_edit_inference(target_model, code_before, eos_token_id_tensor, input_ids , max_token = 4096, temperature= 0.2, top_p = 0.95,top_k = 5,
                             save_prompt_logits: bool = False):
    precode = tokenizer.encode(code_before, add_special_tokens=False, return_tensors="pt").to(target_model.device)
    with contexttimer.Timer() as t:
        with torch.no_grad():
            outputs = efficient_edit_speculative_sampling(
                prefix = input_ids, 
                precode = precode, 
                target_model = target_model,
                eos_token_id_tensor = eos_token_id_tensor,
                max_len=1500 ,
                policy = "greedy",
                temperature = temperature, 
                top_k= top_k, 
                top_p = top_p)
    time = t.elapsed
    tokens = outputs.shape[-1] - input_ids.shape[-1]
    result = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
    return {"time":time, "tokens":tokens, "rate": tokens/time,"result":result}
def autoregressive_inference(model, input_ids, max_token, eos_token_id_tensor, temperature= 0.2, top_p = 0.95,top_k = 5):
    with contexttimer.Timer() as t:
        with torch.no_grad():
            ####kv cache###
            # outputs = model.generate(
            #     input_ids,
            #     max_new_tokens=2048,
            #     temperature=0,
            #     do_sample=False,
            #     eos_token_id=[84274,73594,9902,13874,41233,54275,151645]
            # )
            outputs = autoregressive_sampling(
                x = input_ids, 
                model = model,
                N = max_token ,
                eos_token_id_tensor = eos_token_id_tensor,
                temperature = temperature,
                top_k = top_k,
                top_p = top_p)

    time = t.elapsed
    tokens = outputs.shape[-1] - input_ids.shape[-1]
    result = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=False)
    return {"time":time, "tokens":tokens, "rate": tokens/time,"result":result}


def cleanup_inference_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()

def truncate_string_by_tokens(text, max_tokens=15000):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)

    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        # 将截断后的 token 列表解码回字符串
        return encoding.decode(truncated_tokens)
    
    return text

def cross_prompt(entry: Dict,k) -> str:

#     # return entry['lcz_prompt']
#     # return entry['repoformer_prompt']
#     # return entry['zip_prompt']
    prompt = entry['zip_prompt']
    # return prompt
    # prompt = entry['lcz_prompt']
    input_code = entry["input"]
    if _count_text_tokens(prompt)>10000:
        prompt = truncate_string_by_tokens(prompt)
        prompt+=f"""```
The code to be completed is:
```python
{input_code}
```
"""
    return prompt





    input_code = entry["input"]
    cross_blocks = [item.get("code_block", "") for item in entry.get("cross_file", []) if isinstance(item, dict)][:k]

    total_tokens = sum(_count_text_tokens(block) for block in cross_blocks)
    if total_tokens > MAX_CROSS_FILE_TOKENS:
        cross_blocks = _truncate_blocks_to_budget(cross_blocks, MAX_CROSS_FILE_TOKENS)

    cross_file = _CROSS_FILE_SEP.join(cross_blocks)
    prompt = f"""Please complete the function code based on the context.
The contexts from relevant code fragments from other files of the repo:
```python
{cross_file}
```

The code to be completed is:
```python
{input_code}
```
"""
    return prompt


def _normalize_resume_item(item: Dict[str, Any]) -> str:
    normalized = dict(item)
    for transient_key in (
        "time",
        "tokens",
        "rate",
        "completion",
        "completions",
        "result",
        "results",
        "generate_results",
    ):
        normalized.pop(transient_key, None)
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True)


def _get_resume_key(item: Dict[str, Any]) -> str:
    for key in ("namespace", "_id", "id"):
        value = item.get(key)
        if value is not None:
            return f"{key}:{value}"
    return f"normalized:{_normalize_resume_item(item)}"


def _load_resume_state(data: List[Dict[str, Any]], output_file: Path) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not output_file.exists():
        return [], data

    existing_output = read_json(output_file, True) or []
    if not existing_output:
        return [], data

    data_keys = {_get_resume_key(item) for item in data}
    completed_keys = set()
    resumed_output: List[Dict[str, Any]] = []

    for item in existing_output:
        resume_key = _get_resume_key(item)
        if resume_key not in data_keys:
            raise ValueError(
                f"Found output sample not present in data_file: {resume_key}. "
                f"Please clean {output_file} before resuming."
            )
        if resume_key in completed_keys:
            continue
        completed_keys.add(resume_key)
        resumed_output.append(item)

    remaining_data = [item for item in data if _get_resume_key(item) not in completed_keys]
    return resumed_output, remaining_data

if __name__ == '__main__':
    args = get_parser()
    
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    target_model = AutoModelForCausalLM.from_pretrained(args.target_model,torch_dtype=torch.float16,device_map="auto")
    # draft_model = AutoModelForCausalLM.from_pretrained(args.draft_model, torch_dtype=torch.float16,device_map="auto")

    if 'qwen' in str(args.target_model):
        eos_token_id_tensor = torch.tensor([84274,73594,9902,13874,41233,54275,151645]).to(target_model.device)
    else:
        eos_token_id_tensor = torch.tensor([10252,32021]).to(target_model.device)

    target_model.eval()

    data = read_json(args.data_file,False)
    result, data = _load_resume_state(data or [], args.output_file)
    if result:
        print(f"Loaded {len(result)} completed samples from {args.output_file}")
    if not data:
        print("No remaining samples to run.")
        raise SystemExit(0)

    # data = data[:10]
    time = 0
    k = 10
    num = 0
    for item in tqdm(data):
        sample_result = None
        prompt = None
        input_ids = None
        try:
            prompt = code_generation_prompt(cross_prompt(item,k)).replace('\t', '    ')
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(target_model.device)
            num+=1
            # if input_ids.shape[-1]>20000:
            #     continue
            if args.inference_type =='ar':
            ###AR###
                sample_result = autoregressive_inference(target_model, input_ids, 512, eos_token_id_tensor, temperature=0)
            else:
            ###PD###
                item['map']['drafts'] = [t.replace('\t','    ') for t in item['map']['drafts']]
                draft_code = find_and_slice(item['input'].replace('\t','    '),item['map']['drafts'])
                sample_result = efficient_generation_inference(target_model = target_model, code_before= draft_code, eos_token_id_tensor = eos_token_id_tensor, input_ids = input_ids,max_token = 512, temperature=0)
            item['time'] = sample_result['time']
            item['tokens'] = sample_result['tokens']
            item['rate'] =  sample_result['tokens']/item['time']
            time+=item['time']
            result.append(item)
            save_json(args.output_file,result)
        finally:
            prompt = None
            input_ids = None
            sample_result = None
            cleanup_inference_memory()
    print(time/num)
