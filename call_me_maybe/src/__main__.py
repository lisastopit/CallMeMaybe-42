"""Main entry point for the function calling tool.

Usage:
    uv run python -m src [--functions_definition <path>] [--input <path>] [--output <path>]
"""

import argparse
import sys
from typing import List

from .function_caller import FunctionCaller
from .io_handler import load_function_definitions, load_prompts, save_results
from .models import FunctionCall, FunctionDefinition, Prompt
from .vocabulary import Vocabulary

DEFAULT_FUNCTIONS_DEFINITION = "data/input/functions_definition.json"
DEFAULT_INPUT = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT = "data/output/function_calls.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Function calling tool using constrained decoding with LLMs."
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default=DEFAULT_FUNCTIONS_DEFINITION,
        help=f"Path to function definitions JSON (default: {DEFAULT_FUNCTIONS_DEFINITION})",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help=f"Path to input prompts JSON (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Path for output JSON results (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> int:
    """Run the function calling pipeline.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    args = parse_args()

    print("=" * 60)
    print("  Call Me Maybe - Function Calling Tool")
    print("=" * 60)

    # ---- Load function definitions ----
    print(f"\n[1/4] Loading function definitions from '{args.functions_definition}'...")
    functions: List[FunctionDefinition]
    try:
        functions = load_function_definitions(args.functions_definition)
        print(f"  Loaded {len(functions)} function(s):")
        for fn in functions:
            params = ", ".join(
                f"{n}: {t.type}" for n, t in fn.parameters.items()
            )
            print(f"    - {fn.name}({params})")
    except (FileNotFoundError, ValueError) as e:
        print(f"  [ERROR] {e}", file=sys.stderr)
        return 1

    # ---- Load input prompts ----
    print(f"\n[2/4] Loading prompts from '{args.input}'...")
    prompts: List[Prompt]
    try:
        prompts = load_prompts(args.input)
        print(f"  Loaded {len(prompts)} prompt(s).")
    except (FileNotFoundError, ValueError) as e:
        print(f"  [ERROR] {e}", file=sys.stderr)
        return 1

    if not prompts:
        print("  [WARNING] No valid prompts found. Nothing to process.")
        save_results([], args.output)
        return 0

    # ---- Load model and vocabulary ----
    print("\n[3/4] Initializing LLM model and vocabulary...")
    try:
        from llm_sdk import Small_LLM_Model
        print("  Loading LLM model (Qwen/Qwen3-0.6B)...")
        model = Small_LLM_Model()
        print("  Model loaded successfully.")
        vocab_path: str = model.get_path_to_vocab_file()
        print(f"  Loading vocabulary from '{vocab_path}'...")
        vocab = Vocabulary(vocab_path)
        print(f"  Vocabulary loaded: {len(vocab.token_to_id)} tokens.")
    except Exception as e:
        print(f"  [ERROR] Failed to initialize model/vocabulary: {e}", file=sys.stderr)
        return 1

    # ---- Process prompts ----
    print(f"\n[4/4] Processing {len(prompts)} prompt(s)...")
    caller = FunctionCaller(model, vocab)
    results: List[FunctionCall] = caller.process_all(prompts, functions)

    # ---- Save results ----
    print(f"\n[Done] Saving results to '{args.output}'...")
    try:
        save_results(results, args.output)
    except OSError as e:
        print(f"  [ERROR] Failed to save results: {e}", file=sys.stderr)
        return 1

    print(f"\n  Successfully processed {len(results)}/{len(prompts)} prompts.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
