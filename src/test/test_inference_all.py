from __future__ import annotations

import argparse
from pathlib import Path

from .src.llm_factory import LLMFactory


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "llm_config.toml"


def run_case(name, factory):
    print(f"\n== {name} ==")
    try:
        factory.load()
        result = factory.inference("hello")
        print("ok", result.get("text") if isinstance(result, dict) else result)
        print("latency_sec", factory.last_latency)
        print("tokens", factory.last_token_count)
    except Exception as exc:
        print("failed", exc)
    finally:
        factory.unload()


def run_vllm_batch(factory):
    print("\n== vllm_local batch ==")
    try:
        factory.load()
        result = factory.inference("ignored", batch=["hello", "hi"])
        print("ok", result.get("texts") if isinstance(result, dict) else result)
        print("latency_sec", factory.last_latency)
        print("tokens", factory.last_token_count)
    except Exception as exc:
        print("failed", exc)
    finally:
        factory.unload()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test configured LLM backends.")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to an llm_factory TOML config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    factory = LLMFactory.from_config(str(args.config_path))
    factory.llm_type = "local"
    factory.backend = "local"
    run_case("local", factory)

    factory = LLMFactory.from_config(str(args.config_path))
    factory.llm_type = "api"
    factory.backend = "api"
    run_case("api", factory)

    factory = LLMFactory.from_config(str(args.config_path))
    factory.llm_type = "vllm_local"
    factory.backend = "vllm_local"
    run_case("vllm_local", factory)
    run_vllm_batch(factory)

    factory = LLMFactory.from_config(str(args.config_path))
    factory.llm_type = "vllm_api"
    factory.backend = "vllm_api"
    run_case("vllm_api", factory)


if __name__ == "__main__":
    main()
