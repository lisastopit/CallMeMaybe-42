"""Input/output handling for function calling data files."""

import json
import os
from typing import List

from .models import FunctionCall, FunctionDefinition, Prompt


def load_json_file(path: str) -> object:
    """Load and parse a JSON file, raising descriptive errors on failure.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains invalid JSON.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: '{path}'")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file '{path}': {e}") from e


def load_function_definitions(path: str) -> List[FunctionDefinition]:
    """Load and validate function definitions from a JSON file.

    Args:
        path: Path to the functions definition JSON file.

    Returns:
        List of validated FunctionDefinition objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains invalid JSON or invalid definitions.
    """
    raw: object = load_json_file(path)
    if not isinstance(raw, list):
        raise ValueError(
            f"Function definitions file must contain a JSON array, got {type(raw).__name__}"
        )
    definitions: List[FunctionDefinition] = []
    for i, item in enumerate(raw):
        try:
            definitions.append(FunctionDefinition.model_validate(item))
        except Exception as e:
            raise ValueError(f"Invalid function definition at index {i}: {e}") from e
    if not definitions:
        raise ValueError("Function definitions file is empty — at least one function is required")
    return definitions


def load_prompts(path: str) -> List[Prompt]:
    """Load and validate prompts from a JSON file.

    Args:
        path: Path to the prompts JSON file.

    Returns:
        List of validated Prompt objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains invalid JSON or invalid prompts.
    """
    raw: object = load_json_file(path)
    if not isinstance(raw, list):
        raise ValueError(
            f"Prompts file must contain a JSON array, got {type(raw).__name__}"
        )
    prompts: List[Prompt] = []
    for i, item in enumerate(raw):
        try:
            prompts.append(Prompt.model_validate(item))
        except Exception as e:
            print(f"  [WARNING] Skipping invalid prompt at index {i}: {e}")
    return prompts


def save_results(results: List[FunctionCall], path: str) -> None:
    """Save function call results to a JSON file.

    Args:
        results: List of FunctionCall objects to save.
        path: Output file path.

    Raises:
        OSError: If the output directory cannot be created or the file cannot be written.
    """
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    data = [r.model_dump() for r in results]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Results saved to '{path}'")
