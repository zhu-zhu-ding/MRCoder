from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

LANGUAGE_COMMENT_MAP = {
    "python": "#",
    "c++": "//",
    "java": "//",
    "go": "//",
}


def build_prompt(
    model_type: str,
    cross_file_content: Optional[Union[str, Sequence[Any]]] = None,
    input_code:str = "",
    language: str = "python",
    repo_name: Optional[str] = None,
    file_path: Optional[str] = None,
) -> str:
    model_type = (model_type or "").strip().lower()
    if model_type in {"codellama", "code-llama", "llama"}:
        return _build_codellama_prompt(cross_file_content, language)
    if model_type in {"starcoder", "starcoderbase", "starcoder2", "starcoder-2", "starcoder3", "starcoder-3"}:
        return _build_starcoder_prompt( cross_file_content, language)
    if model_type in {"codeqwen", "qwen", "qwen2.5-coder"}:
        return _build_codeqwen_prompt(cross_file_content,input_code, language, repo_name, file_path)
    if model_type in {"deepseekcoder", "deepseek", "deepseek-coder"}:
        return _build_deepseekcoder_prompt(cross_file_content, language)
    raise ValueError(f"Unsupported model_type: {model_type}")


def _build_codellama_prompt(
    prefix: str,
    suffix: str,
    cross_file_content: Optional[Union[str, Sequence[Any]]],
    language: str,
) -> str:
    pass


def _build_codeqwen_prompt(
    cross_file_content: Optional[Union[str, Sequence[Any]]],
    input_code:str,
    language: str,
    repo_name: Optional[str],
    file_path: Optional[str],
) -> str:
    cross_file = '#-------------------------------------------\n'.join(cross_file_content)
    prompt = f"""Please complete the function code based on the contexts.
The contexts from relevant code fragments from other files of the repo:
The contexts:
```python
{cross_file}
```

The code to be completed is:
```python
{input_code}
```
"""
    return prompt


def _build_starcoder_prompt(
    prefix: str,
    suffix: str,
    cross_file_content: Optional[Union[str, Sequence[Any]]],
    language: str,
) -> str:
    pass


def _build_deepseekcoder_prompt(
    prefix: str,
    suffix: str,
    cross_file_content: Optional[Union[str, Sequence[Any]]],
    language: str,
) -> str:
    pass
