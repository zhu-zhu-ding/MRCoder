from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence
import multiprocessing as mp
import traceback
import hashlib
import math
import random
import gc
random.seed(1234)
import torch
from llm_factory import LLMFactory
from codeutils import CodeUtils
from prompt import build_prompt
from utils import read_json,save_json,get_token_count,extract_code
from time import perf_counter
import re
import tiktoken
from parallel_decoding.speculative_sampling import (
    autoregressive_sampling,
    efficient_generation_speculative_sampling,
)

FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)?\n(.*?)```", re.DOTALL)
MAX_CROSS_FILE_TOKENS = 10000
DEFAULT_SUB_BATCH_SIZE = 64
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


def _chunk_prompts_for_inference(
    prompts: List[str],
    max_batch_size: int,
) -> List[List[str]]:
    if max_batch_size <= 0:
        max_batch_size = 1

    batches: List[List[str]] = []
    for i in range(0, len(prompts), max_batch_size):
        batches.append(prompts[i:i + max_batch_size])
    return batches


def _stable_bucket(token: str, dim: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim




def _run_stage_worker(
    stage: str,
    init_kwargs: Dict[str, Any],
    items: List[Dict[str, Any]],
    conn: Any,
) -> None:
    try:
        coder = SpecletiveCoder(**init_kwargs)
        if stage == "map":
            result = coder._code_map(items)
        elif stage == "reduce":
            result = coder._code_reduce(items)
        else:
            raise ValueError(f"Unsupported stage: {stage}")
        conn.send((True, result))
    except Exception:
        conn.send((False, traceback.format_exc()))
    finally:
        conn.close()


class SpecletiveCoder:
    def __init__(
        self,
        draft_config_path: str,
        target_config_path: str,
        draft_overrides: Optional[Dict[str, Any]] = None,
        target_overrides: Optional[Dict[str, Any]] = None,
        language: str = "python",
        model_type: str = "codeqwen",
        cross_top_k: int = 5,
        partition_method: str = "sequential",
        partition_group_size: int = 4,
        partition_random_seed: int = 42,
        reduce_strategy: str = "pd",
        reduce_top_k: int = 5,
    ) -> None:
        self.language = language
        self.model_type = model_type
        self.cross_top_k = cross_top_k
        self.partition_method = (partition_method or "sequential").strip().lower()
        self.partition_group_size = max(1, int(partition_group_size))
        self.partition_random_seed = partition_random_seed
        self.reduce_strategy = (reduce_strategy or "pd").strip().lower()
        self.reduce_top_k = max(0, int(reduce_top_k))
        self.draft_config_path = draft_config_path
        self.target_config_path = target_config_path
        self.draft_overrides = dict(draft_overrides or {})
        self.target_overrides = dict(target_overrides or {})
        self.model = None
    def _load_model(self,model_type):
        self.model = None
        if model_type=='draft':
            self.model = LLMFactory.from_config(self.draft_config_path)
            if self.draft_overrides:
                self._apply_model_overrides(self.model, self.draft_overrides)
        else:
            self.model = LLMFactory.from_config(self.target_config_path)
            if self.target_overrides:
                self._apply_model_overrides(self.model, self.target_overrides)
        self.model.load()
        return self.model

    def _subprocess_init_kwargs(self) -> Dict[str, Any]:
        return {
            "draft_config_path": self.draft_config_path,
            "target_config_path": self.target_config_path,
            "draft_overrides": self.draft_overrides,
            "target_overrides": self.target_overrides,
            "language": self.language,
            "model_type": self.model_type,
            "cross_top_k": self.cross_top_k,
            "partition_method": self.partition_method,
            "partition_group_size": self.partition_group_size,
            "partition_random_seed": self.partition_random_seed,
            "reduce_strategy": self.reduce_strategy,
            "reduce_top_k": self.reduce_top_k,
        }

    def _apply_model_overrides(self, model: LLMFactory, overrides: Dict[str, Any]) -> None:
        if model.backend in {"vllm_api", "vllm_local"}:
            model.vllm_kwargs.update(overrides)
            return
        if model.backend == "api":
            model.api_kwargs.update(overrides)
            return
        model.local_kwargs.update(overrides)

    def _get_factory_tokenizer(self, model: LLMFactory) -> Any:
        if model.backend == "vllm_local":
            return model.model.get_tokenizer()
        if model.backend == "local" and isinstance(model.model, dict):
            return model.model["tokenizer"]
        raise ValueError(f"Tokenizer is not available for backend: {model.backend}")

    def _get_factory_model_ref(self, model: LLMFactory) -> str:
        if model.backend in {"vllm_api", "vllm_local"}:
            return str(model.vllm_kwargs.get("model_path") or model.vllm_kwargs.get("model") or "")
        if model.backend == "api":
            return str(model.api_kwargs.get("model") or "")
        return str(model.local_kwargs.get("model_path") or "")

    def _get_target_local_model_and_tokenizer(self) -> tuple[torch.nn.Module, Any]:
        target_factory = self._load_model("target")
        self.target_model = target_factory
        if target_factory.backend != "local":
            raise ValueError(
                f"Reduce strategy '{self.reduce_strategy}' requires target backend 'local', got '{target_factory.backend}'."
            )
        loaded = target_factory.model
        if not isinstance(loaded, dict) or "model" not in loaded or "tokenizer" not in loaded:
            raise ValueError("Target local backend did not return model/tokenizer.")
        return loaded["model"], loaded["tokenizer"]

    def _build_reduce_prompt(self, prompt: str) -> str:
        return (
            "###User:\n"
            "You are a code completion assistant.\n"
            f"{prompt}.\n"
            "Please output the complete function code directly.\n"
            "###Assistant\n"
            "```python\n"
        )

    def _find_and_slice(self, needle: str, haystacks: List[str]) -> str:
        for text in haystacks:
            if needle in text:
                start_index = text.find(needle)
                return text[start_index:]
        return needle

    def _get_eos_token_id_tensor(self, tokenizer: Any, model_ref: str, device: torch.device) -> torch.Tensor:
        model_name = (model_ref or getattr(tokenizer, "name_or_path", "") or "").lower()
        if "qwen" in model_name:
            eos_ids = [84274, 73594, 9902, 13874, 41233, 54275, 151645]
        else:
            eos_ids = [10252, 32021]
        tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
        if isinstance(tokenizer_eos, int) and tokenizer_eos not in eos_ids:
            eos_ids.append(tokenizer_eos)
        return torch.tensor(eos_ids, device=device)

    def _cleanup_reduce_memory(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

    def _run_reduce_with_parallel_decoding(
        self,
        data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        target_model, tokenizer = self._get_target_local_model_and_tokenizer()
        model_ref = self._get_factory_model_ref(self.target_model)
        eos_token_id_tensor = self._get_eos_token_id_tensor(tokenizer, model_ref, target_model.device)
        generation_kwargs = dict(self.target_model.local_kwargs)
        max_new_tokens = int(generation_kwargs.get("max_tokens", 1024))
        temperature = float(generation_kwargs.get("temperature", 0.0))
        top_p = float(generation_kwargs.get("top_p", 1.0))
        top_k = int(generation_kwargs.get("top_k", self.reduce_top_k))
        normalized_strategy = self.reduce_strategy

        try:
            for item in data:
                prompt = self._build_reduce_prompt(item["zip_prompt"]).replace("\t", "    ")
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(target_model.device)
                started_at = perf_counter()
                if normalized_strategy == "ar":
                    outputs = autoregressive_sampling(
                        x=input_ids,
                        model=target_model,
                        N=max_new_tokens,
                        eos_token_id_tensor=eos_token_id_tensor,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )
                elif normalized_strategy == "pd":
                    drafts = [draft.replace("\t", "    ") for draft in item.get("map", {}).get("drafts", [])]
                    draft_code = self._find_and_slice(item["input"].replace("\t", "    "), drafts)
                    precode = tokenizer.encode(
                        draft_code,
                        add_special_tokens=False,
                        return_tensors="pt",
                    ).to(target_model.device)
                    outputs = efficient_generation_speculative_sampling(
                        prefix=input_ids,
                        precode=precode,
                        target_model=target_model,
                        eos_token_id_tensor=eos_token_id_tensor,
                        max_len=max_new_tokens,
                        policy="greedy",
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )
                else:
                    raise ValueError(f"Unsupported reduce_strategy: {self.reduce_strategy}")

                elapsed = perf_counter() - started_at
                new_tokens = outputs.shape[-1] - input_ids.shape[-1]
                decoded = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=False)
                reduce_result = extract_code(decoded)
                item["time"] = elapsed
                item["tokens"] = new_tokens
                item["rate"] = new_tokens / elapsed if elapsed > 0 else 0.0
                item["generate_results"] = [reduce_result]
                item["completion"] = [reduce_result]
        finally:
            if hasattr(self, "target_model") and self.target_model is not None:
                self.target_model.unload()
            self._cleanup_reduce_memory()
        return data

    def _cluster_cross_files(self, blocks: List[str]) -> List[List[str]]:
        return self._partition_kmeans(blocks, self.partition_group_size)

    def _partition_cross_files(self, blocks: List[str]) -> List[List[str]]:
        method = self.partition_method
        if method == "sequential":
            return self._partition_sequential(blocks, self.partition_group_size)
        if method == "random":
            return self._partition_random(blocks, self.partition_group_size)
        if method == "interleaved":
            return self._partition_interleaved(blocks, self.partition_group_size)
        if method in {"clustering", "clustering-based", "kmeans", "k-means"}:
            return self._partition_kmeans(blocks, self.partition_group_size)
        raise ValueError(f"Unsupported partition_method: {self.partition_method}")

    def _partition_sequential(self, blocks: List[str], group_size: int) -> List[List[str]]:
        return [blocks[i:i + group_size] for i in range(0, len(blocks), group_size)]

    def _partition_random(self, blocks: List[str], group_size: int) -> List[List[str]]:
        shuffled = list(blocks)
        rng = random.Random(self.partition_random_seed)
        rng.shuffle(shuffled)
        return self._partition_sequential(shuffled, group_size)

    def _partition_interleaved(self, blocks: List[str], group_size: int) -> List[List[str]]:
        if not blocks:
            return []
        group_count = math.ceil(len(blocks) / group_size)
        return [blocks[start::group_count] for start in range(group_count)]

    def _partition_kmeans(self, blocks: List[str], group_size: int) -> List[List[str]]:
        if not blocks:
            return []

        cluster_count = min(len(blocks), math.ceil(len(blocks) / group_size))
        if cluster_count <= 1:
            return [list(blocks)]

        try:
            from sklearn.cluster import KMeans
        except Exception:
            return self._partition_sequential(blocks, group_size)

        vectors = [self._block_to_vector(block) for block in blocks]
        model = KMeans(
            n_clusters=cluster_count,
            random_state=self.partition_random_seed,
            n_init=10,
        )
        labels = model.fit_predict(vectors)

        grouped: List[List[tuple[int, str]]] = [[] for _ in range(cluster_count)]
        for idx, (label, block) in enumerate(zip(labels, blocks)):
            grouped[int(label)].append((idx, block))

        result: List[List[str]] = []
        for cluster in grouped:
            if not cluster:
                continue
            cluster.sort(key=lambda item: item[0])
            result.append([block for _, block in cluster])
        return result

    def _block_to_vector(self, block: str, dim: int = 128) -> List[float]:
        vector = [0.0] * dim
        for token in _FALLBACK_TOKEN_RE.findall(block or ""):
            bucket = _stable_bucket(token, dim)
            vector[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0:
            vector = [value / norm for value in vector]
        return vector

    def _run_stage_subprocess(
        self,
        stage: str,
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        process = ctx.Process(
            target=_run_stage_worker,
            args=(stage, self._subprocess_init_kwargs(), items, child_conn),
        )
        process.start()
        child_conn.close()
        try:
            ok, payload = parent_conn.recv()
        except EOFError as exc:
            process.join()
            raise RuntimeError(
                f"{stage} subprocess exited with code {process.exitcode} without returning a result."
            ) from exc
        finally:
            parent_conn.close()
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"{stage} subprocess exited with code {process.exitcode}.")
        if not ok:
            raise RuntimeError(f"{stage} subprocess failed:\n{payload}")
        return payload

    def run(self, items: List[Dict[str, Any]],save_path: str):
        item_count = len(items)
        map_start = perf_counter()
        items = self._run_stage_subprocess("map", items)
        map_time = (perf_counter() - map_start) / item_count if item_count else 0.0
        for item in items:
            item["map_time"] = map_time
        print(f"-------------------------map_time:{map_time}---------------------------")
        save_json(save_path+'map_result.jsonl', items,False)
        reduce_start = perf_counter()
        items = self._run_stage_subprocess("reduce", items)
        reduce_time = (perf_counter() - reduce_start) / item_count if item_count else 0.0
        print(f"-------------------------map_time:{map_time}---------------------------")
        print(f"-------------------------reduce_time:{reduce_time}---------------------------")
        print(f"-------------------------all_time:{map_time+reduce_time}---------------------------")
        for item in items:
            item["map_time"] = map_time
            item["reduce_time"] = reduce_time
        save_json(save_path, items,False)
        return items
    
    def _code_map(
        self,
        data
    ) -> List[Dict[str, Any]]:
        print("-------------------------begin mapping---------------------------")
        map_prompt = []
        group_counts = []
        for item in data:
            input_code = item['input']
            pre_crossfile_list = [t['code_block'] for t in item['cross_file']][:self.cross_top_k]
            crossfile_list = self._partition_cross_files(pre_crossfile_list)
            group_counts.append(len(crossfile_list))
            item['map'] = {}
            item['map']['cross_file'] = crossfile_list
            for cluster in crossfile_list:
                map_prompt.append(build_prompt(
                    model_type=self.model_type,
                    cross_file_content=cluster,
                    input_code=input_code,
                    language=self.language,
                ))

        if map_prompt:
            entropies, map_result = self._draft_batch_generate(map_prompt)
        else:
            entropies, map_result = [], []

        offset = 0
        results = []
        entropy_groups = []
        for group_count in group_counts:
            results.append(map_result[offset:offset + group_count])
            entropy_groups.append(entropies[offset:offset + group_count])
            offset += group_count

        for item, result, entropy in zip(data, results, entropy_groups):
            item['map']['drafts'] = [extract_code(r) for r in result]
            item['selected_blocks'] = self._filter_with_codeutils(item['map'])
            item['zip_prompt'] = build_prompt(
                    model_type=self.model_type,
                    cross_file_content=item['selected_blocks'],
                    input_code=item['input'],
                    language=self.language,
            )
            item['pre_prompt'] = build_prompt(
                    model_type=self.model_type,
                    cross_file_content=[t['code_block'] for t in item['cross_file']][:self.cross_top_k],
                    input_code=item['input'],
                    language=self.language,
            )
        print("-------------------------before avg token---------------------------")
        print(get_token_count([item['pre_prompt'] for item in data]))
        print("-------------------------after avg token---------------------------")
        print(get_token_count([item['zip_prompt'] for item in data]))
        print("-------------------------mapping end---------------------------")

        return data

    def _code_reduce(
        self,
        data
    ) -> List[Dict[str, Any]]:
        print("-------------------------begin reduce---------------------------")
        data = self._run_reduce_with_parallel_decoding(data)

        print("-------------------------reduce end---------------------------")
        return data

    def _target_batch_generate(self, prompts: List[str]) -> tuple[List[Any], List[str]]:
        
        self.target_model = self._load_model('target')
        tokenizer = self._get_factory_tokenizer(self.target_model)
        prompts = [
        tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "You are a code completion assistant."},
                    {"role": "user", "content": entry},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for entry in prompts
        ]
        DEFAULT_SUB_BATCH_SIZE = 16
        batch_size = self.target_overrides.get("max_num_seqs", DEFAULT_SUB_BATCH_SIZE)
        if not isinstance(batch_size, int) or batch_size <= 0:
            batch_size = DEFAULT_SUB_BATCH_SIZE

        prompt_batches = _chunk_prompts_for_inference(
            prompts,
            max_batch_size=batch_size,
        )

        entropies: List[Any] = []
        texts: List[str] = []
        try:
            for prompt_batch in prompt_batches:
                result = self.target_model.inference("ignored", batch=prompt_batch)
                if isinstance(result, dict) and "texts" in result:
                    entropies.extend(result.get("entropies") or [])
                    texts.extend(result["texts"])
                    continue
                if isinstance(result, dict) and "text" in result:
                    texts.append(result["text"])
                    continue
                if isinstance(result, list):
                    texts.extend(str(item) for item in result)
                    continue
                texts.append(str(result))
        finally:
            self.target_model.unload()

        return entropies, texts
    
    def _draft_batch_generate(self, prompts: List[str]) -> tuple[List[Any], List[str]]:
        self.draft_model = self._load_model('draft')
        tokenizer = self._get_factory_tokenizer(self.draft_model)
        prompts = [
        tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "You are a code completion assistant."},
                    {"role": "user", "content": entry},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for entry in prompts
        ]
        batch_size = self.draft_overrides.get("max_num_seqs", DEFAULT_SUB_BATCH_SIZE)
        if not isinstance(batch_size, int) or batch_size <= 0:
            batch_size = DEFAULT_SUB_BATCH_SIZE

        prompt_batches = _chunk_prompts_for_inference(
            prompts,
            max_batch_size=batch_size,
        )

        entropies: List[Any] = []
        texts: List[str] = []
        try:
            for prompt_batch in prompt_batches:
                result = self.draft_model.inference("ignored", batch=prompt_batch)
                if isinstance(result, dict) and "texts" in result:
                    entropies.extend(result.get("entropies") or [])
                    texts.extend(result["texts"])
                    continue
                if isinstance(result, dict) and "text" in result:
                    texts.append(result["text"])
                    continue
                if isinstance(result, list):
                    texts.extend(str(item) for item in result)
                    continue
                texts.append(str(result))
        finally:
            self.draft_model.unload()

        return entropies, texts

    def _filter_with_codeutils(self, mapped: Dict[str, Any]) -> Dict[str, Any]:
        utils = CodeUtils()
        
        cross_blocks = mapped['cross_file']
        drafts = mapped['drafts']
        selected_blocks = []
        for draft,cross_block in zip(drafts,cross_blocks):
            selected_blocks += utils.filter_by_middle(draft,cross_block)

        total_tokens = sum(_count_text_tokens(block) for block in selected_blocks)
        if total_tokens > MAX_CROSS_FILE_TOKENS:
            selected_blocks = _truncate_blocks_to_budget(selected_blocks, MAX_CROSS_FILE_TOKENS)


        return selected_blocks
