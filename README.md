# MRCoder

## About

MRCoder is a repository-level code completion pipeline with a map-reduce style workflow.

In the `map` stage, the draft model reads grouped cross-file context and produces draft completions.  
In the `reduce` stage, the target model completes the final code with either:

- `pd`: parallel decoding based speculative generation
- `ar`: standard autoregressive decoding

The main entry is [src/run_specletivecoder.py](/home/wpd/workspace/ase26-sepccoder/github_version/src/run_specletivecoder.py).

## Install

```bash
conda env create -f environment.yml
conda activate vllm
```

or

```bash
pip install -r requirements.txt
```

Reference dependencies:

```text
python=3.10
vllm==0.6.0
torch==2.4.0
transformers==4.44.2
tiktoken==0.12.0
openai==2.24.0
numpy==1.26.4
rank-bm25==0.2.2
tree-sitter==0.22.3
tree-sitter-python==0.21.0
codebleu==0.7.0
tomli==2.4.0
```

## Repository Structure

```text
github_version/
├── benchmark/
├── src/
│   ├── specletivecoder.py
│   ├── run_specletivecoder.py
│   ├── llm_factory.py
│   ├── llm_config.toml
│   ├── llm_config_target.toml
│   ├── prompt.py
│   ├── codeutils.py
│   ├── utils.py
│   ├── parallel_decoding/
│   ├── run_infer/
│   └── test/
├── environment.yml
├── requirements.txt
└── README.md
```

## Data Format

The default input file is:

```bash
benchmark/generation/DevEval-main/data/deveval_bm25.jsonl
```

Each JSONL record is expected to contain at least:

- `input`: the code snippet to complete
- `cross_file`: retrieved cross-file context

Each `cross_file` item should normally contain:

- `code_block`: code content from another file

## Configuration

The pipeline uses two config files:

- [src/llm_config.toml](/home/wpd/workspace/ase26-sepccoder/github_version/src/llm_config.toml): draft model config
- [src/llm_config_target.toml](/home/wpd/workspace/ase26-sepccoder/github_version/src/llm_config_target.toml): target model config

### Draft Config: `src/llm_config.toml`

This file controls the `map` stage draft model backend and inference behavior.

```toml
[llm_factory]
llm_type = "vllm_local"
use_api = false
use_vllm = true
use_local = false
vllm_use_api = false
```

Parameter description:

- `llm_type`: backend type. Supported values are `api`, `vllm_api`, `vllm_local`, `local`, `vllm`.
- `use_api`: whether to enable API backend.
- `use_vllm`: whether to enable vLLM backend.
- `use_local`: whether to enable local HuggingFace backend.
- `vllm_use_api`: if `llm_type = "vllm"`, choose between `vllm_api` and `vllm_local`.

Current default behavior:

- the draft model uses `vllm_local`
- the target model uses `local`

Unified inference parameters:

```toml
[llm_factory.infer]
temperature = 0.0
top_p = 1.0
max_tokens = 1024
stop = []
seed = 42
timeout_sec = 180
```

Parameter description:

- `temperature`: decoding temperature.
- `top_p`: nucleus sampling threshold.
- `max_tokens`: maximum number of generated tokens.
- `stop`: stop strings, mainly for API style backends.
- `seed`: random seed for supported backends.
- `timeout_sec`: timeout for API requests.

API backend parameters:

```toml
[llm_factory.api_kwargs]
model = "${OPENAI_MODEL}"
api_key = "${OPENAI_API_KEY}"
base_url = "${OPENAI_BASE_URL}"
```

Parameter description:

- `model`: remote model name.
- `api_key`: API token.
- `base_url`: API endpoint.

vLLM backend parameters:

```toml
[llm_factory.vllm_kwargs]
api_key = "${VLLM_API_KEY}"
base_url = "${VLLM_BASE_URL}"
model = "${VLLM_API_MODEL}"
batch = []
model_path = "${DRAFT_MODEL_PATH}"
trust_remote_code = true
tensor_parallel_size = 1
gpu_memory_utilization = 0.9
max_model_len = 16000
```

Parameter description:

- `api_key`: vLLM API token when using `vllm_api`.
- `base_url`: vLLM API endpoint when using `vllm_api`.
- `model`: remote vLLM model name when using `vllm_api`.
- `batch`: optional prompt batch container used by local vLLM inference.
- `model_path`: local draft model path when using `vllm_local`.
- `trust_remote_code`: whether to trust custom model code from HuggingFace.
- `tensor_parallel_size`: tensor parallel world size for vLLM.
- `gpu_memory_utilization`: GPU memory fraction allowed for vLLM.
- `max_model_len`: maximum model context length.

Local HuggingFace backend parameters:

```toml
[llm_factory.local_kwargs]
model_path = "${DRAFT_MODEL_PATH}"
tokenizer_path = "${DRAFT_TOKENIZER_PATH}"
revision = ""
trust_remote_code = true
device = "cuda"
dtype = "auto"
```

Parameter description:

- `model_path`: local model checkpoint path.
- `tokenizer_path`: local tokenizer path. If empty in your shell setup, usually point it to the same directory as the model.
- `revision`: optional HuggingFace revision.
- `trust_remote_code`: whether to trust model repository custom code.
- `device`: target device such as `cuda`, `cpu`, or `auto`.
- `dtype`: model dtype such as `auto`, `float16`, `bfloat16`, `float32`.

Token counting parameters:

```toml
[llm_factory.local_kwargs.token_count]
enabled = true
backend = "hf"
```

Parameter description:

- `enabled`: whether to enable local token counting.
- `backend`: token counting backend. Current default is `hf`.

### Required Environment Variables

For the current default setup:

```bash
export DRAFT_MODEL_PATH=/path/to/draft-model
export DRAFT_TOKENIZER_PATH=/path/to/draft-tokenizer
export TARGET_MODEL_PATH=/path/to/target-model
export TARGET_TOKENIZER_PATH=/path/to/target-tokenizer
```

If tokenizer and model are in the same directory, you can usually set:

```bash
export DRAFT_TOKENIZER_PATH=$DRAFT_MODEL_PATH
export TARGET_TOKENIZER_PATH=$TARGET_MODEL_PATH
```

If you switch to API backends, also set:

```bash
export OPENAI_MODEL=your-model-name
export OPENAI_API_KEY=your-api-key
export OPENAI_BASE_URL=your-base-url
```

If you switch to `vllm_api`, also set:

```bash
export VLLM_API_MODEL=your-vllm-model
export VLLM_API_KEY=your-vllm-key
export VLLM_BASE_URL=http://your-server:8000/v1
```

## Inference

### Run `run_specletivecoder.py`

The main script is:

```bash
python src/run_specletivecoder.py
```

Default behavior:

- reads `benchmark/generation/DevEval-main/data/deveval_bm25.jsonl`
- uses [src/llm_config.toml](/home/wpd/workspace/ase26-sepccoder/github_version/src/llm_config.toml) as draft config
- uses [src/llm_config_target.toml](/home/wpd/workspace/ase26-sepccoder/github_version/src/llm_config_target.toml) as target config
- writes results to `outputs/speculative_results.jsonl`

### Quick Start

```bash
cd /home/wpd/workspace/ase26-sepccoder/github_version
conda activate vllm

export DRAFT_MODEL_PATH=/path/to/draft-model
export DRAFT_TOKENIZER_PATH=$DRAFT_MODEL_PATH
export TARGET_MODEL_PATH=/path/to/target-model
export TARGET_TOKENIZER_PATH=$TARGET_MODEL_PATH

python src/run_specletivecoder.py
```

### Run with PD Reduce

```bash
python src/run_specletivecoder.py \
  --data-path benchmark/generation/DevEval-main/data/deveval_bm25.jsonl \
  --draft-config src/llm_config.toml \
  --target-config src/llm_config_target.toml \
  --output-dir outputs \
  --output-name speculative_pd.jsonl \
  --language python \
  --model-type codeqwen \
  --cross-top-k 7 \
  --reduce-strategy pd \
  --reduce-top-k 5
```

### Run with AR Reduce

```bash
python src/run_specletivecoder.py \
  --data-path benchmark/generation/DevEval-main/data/deveval_bm25.jsonl \
  --draft-config src/llm_config.toml \
  --target-config src/llm_config_target.toml \
  --output-dir outputs \
  --output-name speculative_ar.jsonl \
  --language python \
  --model-type codeqwen \
  --cross-top-k 7 \
  --reduce-strategy ar \
  --reduce-top-k 5
```

### Argument Description

`run_specletivecoder.py` supports the following arguments:

- `--data-path`: input JSONL benchmark file.
- `--draft-config`: path to draft model TOML config.
- `--target-config`: path to target model TOML config.
- `--output-dir`: output directory.
- `--output-name`: output JSONL file name.
- `--language`: programming language, default `python`.
- `--model-type`: prompt template family such as `codeqwen` or `deepseek`.
- `--cross-top-k`: number of retrieved cross-file blocks kept before the map stage.
- `--reduce-strategy`: reduce-stage decoding strategy, one of `pd` or `ar`.
- `--reduce-top-k`: top-k used in reduce-stage decoding.

### Output Files

After running the pipeline, you should get:

- final output: `outputs/<output-name>`
- map stage intermediate file: `outputs/<output-name>map_result.jsonl`

## Baseline

The repository also contains a standalone vLLM baseline:

```bash
bash src/run_infer/run_vllm.sh
```

or

```bash
python src/run_infer/run_vllm_batch_inference.py \
  --model-path /path/to/target-model \
  --input-file benchmark/generation/DevEval-main/data/deveval_bm25.jsonl \
  --output-dir outputs/baseline \
  --output-name baseline_completion.jsonl
```

## Parallel Decoding

The low-level parallel decoding implementation is under:

- [src/parallel_decoding/inference.py](/home/wpd/workspace/ase26-sepccoder/github_version/src/parallel_decoding/inference.py)
- [src/parallel_decoding/speculative_sampling.py](/home/wpd/workspace/ase26-sepccoder/github_version/src/parallel_decoding/speculative_sampling.py)

`run_specletivecoder.py` already integrates:

- AR reduce from `autoregressive_sampling`
- PD reduce from `efficient_generation_speculative_sampling`

## Notes

- The main pipeline assumes `cross_file` retrieval has already been done in the input JSONL.
- The draft stage still defaults to `vllm_local`.
- The target stage currently defaults to local HuggingFace inference for AR and PD reduce.
- The intermediate map result file is written by directly appending `map_result.jsonl` to the output path string.
