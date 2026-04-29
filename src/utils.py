import json
import logging
import os
import subprocess
import threading
import tiktoken
from typing import List
from concurrent.futures import ThreadPoolExecutor
def extract_code(completion: str) -> str:
    import re
    import textwrap

    if not completion:
        return ""

    def _clean(text: str) -> str:
        text = re.sub(r"```python\s*", "", text, flags=re.IGNORECASE)
        text = text.replace("```", "")
        return text.strip()

    # 1. 优先提取第一个 ```python ... ``` 代码块
    python_fence_re = re.compile(
        r"```python\s*\n?(.*?)```",
        re.IGNORECASE | re.DOTALL,
    )
    m = python_fence_re.search(completion)
    if m:
        return _clean(m.group(1))

    # 2. 没有 fenced python 代码块时，按第一个 def / async def 提取
    text = _clean(completion)
    lines = text.splitlines()

    def_re = re.compile(
        r"^([ \t]*)(?:async[ \t]+)?def[ \t]+[A-Za-z_]\w*[ \t]*\("
    )

    start = None
    base_indent = 0

    for i, line in enumerate(lines):
        m = def_re.match(line)
        if m:
            start = i
            base_indent = len(m.group(1).expandtabs(4))

            # 如果 def 上面紧挨着 decorator，也一起带上
            j = i - 1
            while j >= 0:
                prev = lines[j]
                if not prev.strip():
                    break
                prev_indent = len(prev[: len(prev) - len(prev.lstrip(" \t"))].expandtabs(4))
                if prev.lstrip().startswith("@") and prev_indent == base_indent:
                    start = j
                    j -= 1
                    continue
                break
            break

    if start is None:
        return text

    end = len(lines)
    header_done = False
    paren_depth = 0
    def_line_found = False

    for i in range(start, len(lines)):
        line = lines[i]
        stripped = line.strip()

        if not def_line_found and def_re.match(line):
            def_line_found = True

        if def_line_found and not header_done:
            paren_depth += line.count("(") - line.count(")")
            if stripped.endswith(":") and paren_depth <= 0:
                header_done = True
            continue

        if header_done and stripped:
            indent = len(line[: len(line) - len(line.lstrip(" \t"))].expandtabs(4))
            if indent <= base_indent:
                end = i
                break

    return _clean(textwrap.dedent("\n".join(lines[start:end])))


# def extract_code(completion: str) -> str:
#     import json
#     import re

#     FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)?\n(.*?)```", re.DOTALL)
#     if not completion:
#         return ""
#     m = FENCE_RE.findall(completion)
#     if m:
#         parts = [p.strip("\n") for p in m if p.strip()]
#         return "\n\n".join(parts).strip()
#     return completion.replace("```python",'').strip()


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

def save_json(data_path,data_list,is_list=True):
    if is_list:
        try:
            open(data_path, 'w', encoding="utf-8",).write(json.dumps(data_list,indent=4))
        except Exception as e:
            print(f"save json_path {data_path} exception:{e}")
            return None
    else:
        try:
            with open(data_path, 'w', encoding="utf-8") as jsonl_file:
                for save_item in data_list:
                    jsonl_file.write(json.dumps(save_item) + '\n')
        except Exception as e:
            print(f"save json_path {data_path} exception:{e}")
            return None

class _IgnoreDataflowWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "There is no reference data-flows extracted from the whole corpus" not in msg


_root_logger = logging.getLogger()
if not any(isinstance(f, _IgnoreDataflowWarning) for f in _root_logger.filters):
    _root_logger.addFilter(_IgnoreDataflowWarning())


def get_token_count(prompts: List[str], model_name: str = "gpt-5", num_workers: int = 8) -> float:
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base")

    if not prompts:
        return 0.0

    worker_count = max(1, min(num_workers, len(prompts)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        counts = list(executor.map(lambda p: len(encoding.encode(p)), prompts))
    
    total_tokens = sum(counts)
    num_prompts = len(prompts)
    
    return total_tokens / num_prompts
