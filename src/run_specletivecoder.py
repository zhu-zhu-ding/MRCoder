from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List


SRC_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SRC_ROOT.parent
DEFAULT_DATA_PATH = REPO_ROOT / "benchmark/generation/DevEval-main/data/deveval_bm25.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"
DEFAULT_DRAFT_CONFIG = SRC_ROOT / "llm_config.toml"
DEFAULT_TARGET_CONFIG = SRC_ROOT / "llm_config_target.toml"


def _print_maps(title: str, maps: List[Dict[str, Any]]) -> None:
    print(f"\n==== {title} ====")
    for i, m in enumerate(maps):
        print(f"\n-- map {i} --")
        print(m.get("prompt", ""))
        print("\n---")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run speculative code generation.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Input JSONL benchmark file.",
    )
    parser.add_argument(
        "--draft-config",
        type=Path,
        default=DEFAULT_DRAFT_CONFIG,
        help="Draft model config TOML.",
    )
    parser.add_argument(
        "--target-config",
        type=Path,
        default=DEFAULT_TARGET_CONFIG,
        help="Target model config TOML.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store results.",
    )
    parser.add_argument(
        "--output-name",
        default="speculative_results.jsonl",
        help="Output JSONL file name.",
    )
    parser.add_argument(
        "--language",
        default="python",
        help="Programming language of the benchmark.",
    )
    parser.add_argument(
        "--model-type",
        default="codeqwen",
        help="Prompt template family, e.g. codeqwen or deepseek.",
    )
    parser.add_argument(
        "--cross-top-k",
        type=int,
        default=7,
        help="Number of retrieved cross-file blocks kept before map stage.",
    )
    parser.add_argument(
        "--reduce-strategy",
        default="pd",
        choices=("ar", "pd"),
        help="Reduce-stage decoding strategy.",
    )
    parser.add_argument(
        "--reduce-top-k",
        type=int,
        default=5,
        help="Top-k used by reduce-stage AR/PD decoding.",
    )
    return parser.parse_args()


def main() -> None:
    from specletivecoder import SpecletiveCoder

    args = parse_args()
    with args.data_path.open("r", encoding="utf-8") as handle:
        data = [json.loads(line) for line in handle]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name

    coder = SpecletiveCoder(
        draft_config_path=str(args.draft_config),
        target_config_path=str(args.target_config),
        language=args.language,
        model_type=args.model_type,
        cross_top_k=args.cross_top_k,
        reduce_strategy=args.reduce_strategy,
        reduce_top_k=args.reduce_top_k,
    )
    coder.run(data, str(output_path))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
