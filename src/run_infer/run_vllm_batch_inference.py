from __future__ import annotations

import argparse
import json
import os
import sys
import re
from pathlib import Path
from typing import Dict, Iterable, List

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3,1"
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prompt import build_prompt

import argparse
import json
from pathlib import Path

try:
    import tiktoken
except Exception:
    tiktoken = None

FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)?\n(.*?)```", re.DOTALL)
MAX_CROSS_FILE_TOKENS = 12000
_CROSS_FILE_SEP = "#-------------------------------------------\n"
_FALLBACK_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|\S")
_ENCODER = tiktoken.get_encoding("cl100k_base") if tiktoken is not None else None


def extract_code(completion: str) -> str:
    if not completion:
        return ""
    m = FENCE_RE.findall(completion)
    if m:
        parts = [p.strip("\n") for p in m if p.strip()]
        return "\n\n".join(parts).strip()
    return completion.strip()

def get_code_snippet(code_str: str, n: int, mode: str = 'tail') -> str:
    """
    获取代码字符串的前 n 行或后 n 行，保留原始缩进和换行符。
    
    :param code_str: 原始代码字符串
    :param n: 获取的行数
    :param mode: 'head' 表示前 n 行，'tail' 表示后 n 行
    :return: 截取后的字符串
    """
    # splitlines(True) 会保留每行末尾的换行符
    lines = code_str.splitlines(keepends=True)
    
    if mode == 'head':
        selected_lines = lines[:n]
    elif mode == 'tail':
        selected_lines = lines[-n:]
    else:
        raise ValueError("mode 必须是 'head' 或 'tail'")
        
    return "".join(selected_lines)

def build_user_prompt(entry: Dict) -> str:
    input_code = entry['input']
    prompt = f"""Please complete the function code based on the contexts.

The contexts above the function are:
```python
{contexts_above}
```

The contexts below the function are:
```python
{contexts_below}
```

The code to be completed is:
```python
{input_code}
```
"""
    return prompt

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


def cross_prompt(entry: Dict) -> str:
    # return entry['zip']["prompt"]repoformer_prompt
    # return entry["repoformer_prompt"]+'```'
    # return entry["lcz_prompt"]+'```'

    return entry["zip_prompt"]

    input_code = entry["input"]
    cross_blocks = [item.get("code_block", "") for item in entry.get("cross_file", []) if isinstance(item, dict)][:10]

    # total_tokens = sum(_count_text_tokens(block) for block in cross_blocks)
    # if total_tokens > MAX_CROSS_FILE_TOKENS:
    #     cross_blocks = _truncate_blocks_to_budget(cross_blocks, MAX_CROSS_FILE_TOKENS)

    cross_file = _CROSS_FILE_SEP.join(cross_blocks)
    prompt = f"""Please generate the function code based on the context.
The contexts from relevant code fragments from other files of the repo:
```python
{cross_file}
```

The code to generate is:
```python
{input_code}
```
"""
    return prompt



def generate_batches(
    llm: LLM,
    sampling_params: SamplingParams,
    entries: List[Dict],
    default_system_prompt: str,
    lora_request: LoRARequest | None = None,
) -> List[str]:
    tokenizer = llm.get_tokenizer()
    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": default_system_prompt},
                {"role": "user", "content": cross_prompt(entry)},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        # build_user_prompt(entry)
        for entry in entries
    ]
    sampling_params.stop_token_ids = [tokenizer.eos_token_id]
    results = llm.generate(
        prompts,
        sampling_params=sampling_params
    )
    completions = []

    for result in results:
        if not result.outputs:
            completions.append("")
            continue
        completions.append([extract_code(item.text) for item in result.outputs])
    return completions

def iter_jsonl(path: str) -> Iterable[Dict]:
    
    with open(path, "r", encoding="utf-8") as src:
        for line in src:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
def read_json(data_path,is_list=True):
    if is_list:
        try:
            return json.load(open(data_path, 'r',encoding="utf-8"))
        except Exception as e:
            print(f"read json_path {data_path} exception:{e}")
            return None
    else:
        try:
            return [json.loads(line) for line in open(data_path, 'r',encoding="utf-8")]
        except Exception as e:
            print(f"read json_path {data_path} exception:{e}")
            return None


def _get_resume_key(entry: Dict) -> str:
    namespace = entry.get("namespace")
    if namespace is None:
        raise ValueError("Resume check requires every entry to contain a 'namespace' field.")
    return str(namespace)


def _get_resume_state(input_entries: List[Dict], output_path: str) -> tuple[int, str]:
    if not os.path.exists(output_path):
        return 0, "w"

    completed_entries = list(iter_jsonl(output_path))
    if not completed_entries:
        return 0, "w"

    if len(completed_entries) > len(input_entries):
        raise ValueError(
            f"Existing output has {len(completed_entries)} rows, more than input {len(input_entries)} rows: {output_path}"
        )

    for idx, (completed_entry, input_entry) in enumerate(zip(completed_entries, input_entries), start=1):
        if _get_resume_key(completed_entry) != _get_resume_key(input_entry):
            raise ValueError(
                f"Resume check failed at line {idx}: existing output namespace does not match input namespace. "
                f"Please remove or rename {output_path} before rerunning."
            )

    return len(completed_entries), "a"


def run_inference(
    model_path: str,
    input_path: str,
    output_path: str,
    batch_size: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    n:int,
    tensor_parallel_size: int,
    lora_path: str,
    trust_remote_code: bool,
    system_prompt: str,
    eos_token_ids: List[int],
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    input_entries = read_json(input_path, False) or []
    completed_count, write_mode = _get_resume_state(input_entries, output_path)
    remaining_entries = input_entries[completed_count:]

    if completed_count:
        print(f"Found {completed_count} completed rows in {output_path}, resuming from row {completed_count + 1}.")
    if not remaining_entries:
        print(f"No remaining rows to process for {output_path}.")
        return

    llm_kwargs = dict(
        model=model_path,
        max_model_len = 16000,
        gpu_memory_utilization = 0.90,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=trust_remote_code,
    )

    llm = LLM(**llm_kwargs)
    lora_request: LoRARequest | None = None
    if lora_path:
        lora_request = LoRARequest(
            lora_name="inference_adapter",
            lora_int_id=1,
            lora_path=lora_path,
        )
        llm.llm_engine.add_lora(lora_request)
        llm.llm_engine.pin_lora(lora_request.lora_int_id)
    # eos_token_ids = llm.get_tokenizer().eos_token_id
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        # # n = 10,
        # stop_token_ids=eos_token_ids or None,
    )

    batch: List[Dict] = []
    with open(output_path, write_mode, encoding="utf-8") as dst:
        for entry in remaining_entries:
            batch.append(entry)
            if len(batch) < batch_size:
                continue

            completions = generate_batches(
                llm, sampling_params, batch, system_prompt, lora_request
            )
            for entry, completion in zip(batch, completions):
                # entry["generate_results"] = completion
                entry["completion"] = completion
                dst.write(json.dumps(entry, ensure_ascii=False))
                dst.write("\n")
            batch.clear()

        if batch:
            completions = generate_batches(
                llm, sampling_params, batch, system_prompt, lora_request
            )
            for entry, completion in zip(batch, completions):
                # entry["generate_results"] = completion
                entry["completion"] = completion
                dst.write(json.dumps(entry, ensure_ascii=False))
                dst.write("\n")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Batch inference using vLLM.")
    parser.add_argument(
        "--model-path",
        default=os.environ.get("TARGET_MODEL_PATH", ""),
        help="Path to the local huggingface model directory.",
    )
    parser.add_argument(
        "--input-file",
        default=str(repo_root / "benchmark/generation/DevEval-main/data/deveval_bm25.jsonl"),
        help="JSONL file containing prompts.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(repo_root / "outputs/baseline"),
        help="Directory where the JSONL with completions will be stored.",
    )
    parser.add_argument(
        "--output-name",
        default="zip_python_structured_span_completion.jsonl",
        help="Name of the output file, defaults to a descriptive suffix.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of prompts to send per vLLM request.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate per sample.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Top-p nucleus sampling.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size to pass to vLLM.",
    )
    parser.add_argument(
        "--lora-path",
        default="",
        help="Optional LoRA checkpoint path to load.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass through to vLLM for models requiring custom code.",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful assistant.",
        help="System prompt injected into every request unless the entry overrides it.",
    )
    parser.add_argument(
        "--eos-token-ids",
        type=int,
        nargs="*",
        default=[151659, 151661, 151662, 151663, 151664, 151643, 151645],
        help="Token IDs that stop generation; defaults to Qwen3-Coder EOS IDs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_path:
        raise ValueError(
            "Missing model path. Pass --model-path or export TARGET_MODEL_PATH."
        )
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name)
    # args.eos_token_ids = [84274,73594,9902,13874,41233,54275,151645]
    run_inference(
        model_path=args.model_path,
        input_path=args.input_file,
        output_path=output_path,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        n = args.n,
        tensor_parallel_size=args.tensor_parallel_size,
        lora_path=args.lora_path,
        trust_remote_code=args.trust_remote_code,
        system_prompt=args.system_prompt,
        eos_token_ids=args.eos_token_ids,
    )
    print(f"Wrote completions to {output_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
