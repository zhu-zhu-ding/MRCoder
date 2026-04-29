from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
import gc
import os
import time


class LLMFactoryError(RuntimeError):
    pass


@dataclass
class LLMFactory:
    """Factory that routes inference to API, vLLM, or local backends."""

    llm_type: Optional[str] = None
    use_api: bool = False
    use_vllm: bool = False
    use_local: bool = True
    vllm_use_api: bool = False
    api_kwargs: Dict[str, Any] = field(default_factory=dict)
    vllm_kwargs: Dict[str, Any] = field(default_factory=dict)
    local_kwargs: Dict[str, Any] = field(default_factory=dict)
    model: Any = None
    last_latency: Optional[float] = None
    last_usage: Optional[Dict[str, Any]] = None
    last_token_count: Optional[int] = None

    def __post_init__(self) -> None:
        self.backend = _resolve_backend(
            self.llm_type,
            self.use_api,
            self.use_vllm,
            self.use_local,
            self.vllm_use_api,
        )

    def load(self) -> Any:
        """Optional: load or initialize backend-specific model/client if provided by inference.py."""
        loader = _get_loader_fn(self.backend)
        if loader is None:
            return None
        if self.backend == "api":
            self.model = loader(**self.api_kwargs)
        elif self.backend in {"vllm_api", "vllm_local"}:
            self.model = loader(**self.vllm_kwargs)
        else:
            self.model = loader(**self.local_kwargs)
        return self.model

    def unload(self) -> None:
        """Release the loaded model and free any GPU memory owned by this factory."""
        _release_loaded_model(self.model)
        self.model = None

    def inference_api(self, prompt: str, **kwargs: Any) -> Any:
        """Run API inference via inference.py."""
        return self._run_inference(prompt, "api", self.api_kwargs, **kwargs)

    def inference_vllm_api(self, prompt: str, **kwargs: Any) -> Any:
        """Run vLLM API inference via inference.py."""
        return self._run_inference(prompt, "vllm_api", self.vllm_kwargs, **kwargs)

    def inference_vllm_local(self, prompt: str, **kwargs: Any) -> Any:
        """Run vLLM local inference via inference.py."""
        return self._run_inference(prompt, "vllm_local", self.vllm_kwargs, **kwargs)

    def inference_vllm(self, prompt: str, **kwargs: Any) -> Any:
        """Run vLLM inference via inference.py."""
        if self.vllm_use_api:
            return self.inference_vllm_api(prompt, **kwargs)
        return self.inference_vllm_local(prompt, **kwargs)

    def inference_local(self, prompt: str, **kwargs: Any) -> Any:
        """Run local inference via inference.py."""
        return self._run_inference(prompt, "local", self.local_kwargs, **kwargs)

    def inference(self, prompt: str, **kwargs: Any) -> Any:
        """Run inference via the selected backend."""
        if self.backend == "api":
            return self.inference_api(prompt, **kwargs)
        if self.backend == "vllm_api":
            return self.inference_vllm_api(prompt, **kwargs)
        if self.backend == "vllm_local":
            return self.inference_vllm_local(prompt, **kwargs)
        return self.inference_local(prompt, **kwargs)

    def __call__(self, prompt: str, **kwargs: Any) -> Any:
        return self.inference(prompt, **kwargs)

    @classmethod
    def from_config(cls, path: str) -> "LLMFactory":
        config = _load_toml(path).get("llm_factory", {})
        infer_defaults = dict(config.get("infer", {}))
        api_kwargs = _merge_dicts(infer_defaults, config.get("api_kwargs", {}))
        vllm_kwargs = _merge_dicts(infer_defaults, config.get("vllm_kwargs", {}))
        local_kwargs = _merge_dicts(infer_defaults, config.get("local_kwargs", {}))

        return cls(
            llm_type=config.get("llm_type"),
            use_api=bool(config.get("use_api", False)),
            use_vllm=bool(config.get("use_vllm", False)),
            use_local=bool(config.get("use_local", True)),
            vllm_use_api=bool(config.get("vllm_use_api", False)),
            api_kwargs=api_kwargs,
            vllm_kwargs=vllm_kwargs,
            local_kwargs=local_kwargs,
        )

    def _run_inference(
        self,
        prompt: str,
        backend: str,
        backend_kwargs: Dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        fn = _get_inference_fn(backend)
        if fn is None:
            raise LLMFactoryError(
                f"backend '{backend}' inference function is not available."
            )
        payload = {}
        payload.update(backend_kwargs)
        payload.update(kwargs)
        if self.model is not None and "model" not in kwargs:
            payload["model"] = self.model
        start = time.perf_counter()
        result = fn(prompt, **payload)
        self.last_latency = time.perf_counter() - start
        usage = _extract_usage(result)
        self.last_usage = usage or None
        self.last_token_count = _extract_total_tokens(usage)
        if isinstance(result, dict):
            result["latency_sec"] = self.last_latency
            if isinstance(result.get("usage"), dict):
                result["usage"].setdefault("latency_sec", self.last_latency)
        return result


def _resolve_backend(
    llm_type: Optional[str],
    use_api: bool,
    use_vllm: bool,
    use_local: bool,
    vllm_use_api: bool,
) -> str:
    if llm_type:
        backend = llm_type.strip().lower()
        if backend == "vllm":
            return "vllm_api" if vllm_use_api else "vllm_local"
        if backend not in {"api", "vllm_api", "vllm_local", "local"}:
            raise ValueError(f"Unsupported llm_type: {llm_type}")
        return backend
    selected = [name for name, enabled in (
        ("api", use_api),
        ("vllm", use_vllm),
        ("local", use_local),
    ) if enabled]
    if len(selected) != 1:
        raise ValueError(
            f"Exactly one backend must be enabled, got: {selected or 'none'}"
        )
    if selected[0] == "vllm":
        return "vllm_api" if vllm_use_api else "vllm_local"
    return selected[0]


def _get_inference_fn(backend: str) -> Optional[Callable[..., Any]]:
    return _INFERENCE_FNS.get(backend)


def _get_loader_fn(backend: str) -> Optional[Callable[..., Any]]:
    return _LOADER_FNS.get(backend)


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    merged.update(override or {})
    return merged


def _load_toml(path: str) -> Dict[str, Any]:
    try:
        import tomllib
    except Exception:  # pragma: no cover - fallback for py<3.11
        import tomli as tomllib  # type: ignore
    with open(path, "rb") as f:
        return _resolve_config_value(tomllib.load(f))


def _resolve_config_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_config_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_resolve_config_value(item) for item in value]
    if isinstance(value, str):
        expanded = os.path.expandvars(os.path.expanduser(value))
        if expanded.startswith("./") or expanded.startswith("../"):
            return str(Path(expanded))
        return expanded
    return value


def _extract_usage(result: Any) -> Dict[str, Any]:
    if result is None:
        return {}
    if isinstance(result, dict):
        usage = result.get("usage") or result.get("token_usage")
        return usage if isinstance(usage, dict) else {}
    usage = getattr(result, "usage", None)
    return usage if isinstance(usage, dict) else {}


def _extract_total_tokens(usage: Dict[str, Any]) -> Optional[int]:
    if not usage:
        return None
    for key in ("total_tokens", "total", "tokens", "token_count"):
        value = usage.get(key)
        if isinstance(value, int):
            return value
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
        return prompt_tokens + completion_tokens
    return None


def _release_loaded_model(model: Any) -> None:
    if model is None:
        _clear_torch_cuda_cache()
        return

    if isinstance(model, dict):
        for key in tuple(model.keys()):
            _release_loaded_model(model.pop(key, None))
        _clear_torch_cuda_cache()
        return

    if isinstance(model, (list, tuple, set)):
        for value in model:
            _release_loaded_model(value)
        _clear_torch_cuda_cache()
        return

    for attr_name in ("model_executor", "llm_engine", "engine", "worker", "driver_worker"):
        target = getattr(model, attr_name, None)
        if target is not None and target is not model:
            _call_cleanup(target)

    _call_cleanup(model)
    del model
    gc.collect()
    _clear_torch_cuda_cache()


def _call_cleanup(obj: Any) -> None:
    for method_name in ("shutdown", "close", "terminate"):
        method = getattr(obj, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:
                pass


def _clear_torch_cuda_cache() -> None:
    try:
        import torch
    except Exception:
        return

    if not torch.cuda.is_available():
        return

    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def load_local(
    model_path: str,
    tokenizer_path: Optional[str] = None,
    device: str = "cuda",
    dtype: str = "auto",
    trust_remote_code: bool = True,
    revision: Optional[str] = None,
    **_: Any,
) -> Dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tok_path = tokenizer_path or model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tok_path,
        trust_remote_code=trust_remote_code,
        revision=revision,
    )

    torch_dtype = _resolve_dtype(dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        revision=revision,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "auto" else None,
    )
    if device not in {"auto", None}:
        model = model.to(device)
    model.eval()
    return {"model": model, "tokenizer": tokenizer}


def local_inference(
    prompt: str,
    model: Any = None,
    tokenizer: Any = None,
    model_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    device: str = "cuda",
    dtype: str = "auto",
    trust_remote_code: bool = True,
    revision: Optional[str] = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: Optional[bool] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    model, tokenizer = _ensure_local_model(
        model,
        tokenizer,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        revision=revision,
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)

    if do_sample is None:
        do_sample = temperature > 0

    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": do_sample,
    }

    if "stop" in kwargs:
        pass

    output_ids = model.generate(input_ids=input_ids, **gen_kwargs)
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    completion = full_text[len(prompt):] if full_text.startswith(prompt) else full_text

    usage = _make_usage(
        prompt_tokens=len(input_ids[0]),
        completion_tokens=len(output_ids[0]) - len(input_ids[0]),
    )
    return {
        "text": completion,
        "usage": usage,
        "raw_text": full_text,
    }


def load_vllm_local(
    model_path: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None,
    trust_remote_code: bool = True,
    **_: Any,
) -> Any:
    from vllm import LLM

    return LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=128,
        trust_remote_code=trust_remote_code,
    )


def vllm_local_inference(
    prompt: str,
    model: Any = None,
    model_path: Optional[str] = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 128,
    batch: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    if model is None:
        if not model_path:
            raise ValueError("model_path is required for vllm_local_inference")
        model = load_vllm_local(model_path=model_path, **kwargs)

    from vllm import SamplingParams

    params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        logprobs=1,
        stop_token_ids=[151659, 151661, 151662, 151663, 151664, 151643, 151645],
    )
    prompts = list(batch) if batch is not None else [prompt]
    outputs = model.generate(prompts, params)

    texts = []
    entropies = []
    usage_list = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for out in outputs:
        candidate = out.outputs[0]
        text = candidate.text
        texts.append(text)
        prompt_tokens = len(out.prompt_token_ids)
        completion_tokens = len(candidate.token_ids)
        usage_list.append(_make_usage(prompt_tokens, completion_tokens))
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        entropies.append(_sequence_entropy_from_output(candidate))

    usage = _make_usage(total_prompt_tokens, total_completion_tokens)
    if batch is None:
        return {
            "text": texts[0],
            "entropy": entropies[0],
            "usage": usage,
            "raw": outputs[0],
        }
    return {
        "texts": texts,
        "entropies": entropies,
        "usage": usage,
        "usage_list": usage_list,
        "raw": outputs,
    }


def load_vllm_api(**_: Any) -> None:
    return None


def vllm_api_inference(
    prompt: str,
    model: Optional[str] = None,
    base_url: str = "http://localhost:8080/v1",
    api_key: str = "",
    temperature: float = 0.0,
    max_tokens: int = 256,
    timeout_sec: int = 180,
    **_: Any,
) -> Dict[str, Any]:
    return _openai_chat(
        prompt=prompt,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_sec=timeout_sec,
    )


def load_api(**_: Any) -> None:
    return None


def api_inference(
    prompt: str,
    model: Optional[str] = None,
    base_url: str = "https://api.example.com/v1",
    api_key: str = "",
    temperature: float = 0.0,
    max_tokens: int = 256,
    timeout_sec: int = 180,
    **_: Any,
) -> Dict[str, Any]:
    return _openai_chat(
        prompt=prompt,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_sec=timeout_sec,
    )


def _openai_chat(
    prompt: str,
    model: Optional[str],
    base_url: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
) -> Dict[str, Any]:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("openai package is required for API inference") from exc

    if not model:
        raise ValueError("model is required for API inference")

    client = OpenAI(base_url=base_url, api_key=api_key)
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout_sec,
    )
    latency = time.perf_counter() - start
    text = resp.choices[0].message.content
    usage = _extract_openai_usage(resp)
    usage["latency_sec"] = latency
    return {"text": text, "usage": usage, "raw": resp}


def _ensure_local_model(
    model: Any,
    tokenizer: Any,
    **kwargs: Any,
) -> Tuple[Any, Any]:
    if model is not None and tokenizer is not None:
        return model, tokenizer
    if isinstance(model, dict):
        mdl = model.get("model")
        tok = model.get("tokenizer")
        if mdl is not None and tok is not None:
            return mdl, tok
    if isinstance(model, tuple) and len(model) == 2:
        return model[0], model[1]
    if not kwargs.get("model_path"):
        raise ValueError("model_path is required for local_inference")
    loaded = load_local(**kwargs)
    return loaded["model"], loaded["tokenizer"]


def _resolve_dtype(dtype: str):
    import torch

    if dtype in ("auto", None):
        return None
    if isinstance(dtype, str):
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if dtype.lower() in mapping:
            return mapping[dtype.lower()]
    return None


def _make_usage(prompt_tokens: int, completion_tokens: int) -> Dict[str, int]:
    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(prompt_tokens + completion_tokens),
    }


def _extract_openai_usage(resp: Any) -> Dict[str, int]:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return {}


def _sequence_entropy_from_output(output: Any) -> float:
    token_count = len(getattr(output, "token_ids", []) or [])
    if token_count == 0:
        return 0.0

    cumulative_logprob = getattr(output, "cumulative_logprob", None)
    if isinstance(cumulative_logprob, (float, int)):
        # Average token-level entropy / negative log-likelihood.
        return float(-cumulative_logprob / token_count)

    logprobs = getattr(output, "logprobs", None) or []
    nll_sum = 0.0
    valid_steps = 0
    for step in logprobs:
        if not step:
            continue
        best = None
        for value in step.values():
            lp = getattr(value, "logprob", None)
            if isinstance(lp, (float, int)):
                if best is None or lp > best:
                    best = float(lp)
        if best is None:
            continue
        nll_sum += -best
        valid_steps += 1

    return float(nll_sum / valid_steps) if valid_steps > 0 else 0.0


_INFERENCE_FNS: Dict[str, Callable[..., Any]] = {
    "api": api_inference,
    "vllm_api": vllm_api_inference,
    "vllm_local": vllm_local_inference,
    "local": local_inference,
}

_LOADER_FNS: Dict[str, Callable[..., Any]] = {
    "api": load_api,
    "vllm_api": load_vllm_api,
    "vllm_local": load_vllm_local,
    "local": load_local,
}
