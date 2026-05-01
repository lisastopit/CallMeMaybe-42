"""Function caller: orchestrates LLM calls and constrained decoding.

This module combines:
  1. Prompt construction for both function selection and argument extraction.
  2. The constrained decoder to enforce valid structured output.
  3. Type coercion for final output validation.
"""

from typing import Any, Dict, List

from .constrained_decoder import ConstrainedDecoder
from .models import FunctionCall, FunctionDefinition, Prompt
from .vocabulary import Vocabulary


def _build_function_selection_prompt(
    prompt: str,
    functions: List[FunctionDefinition],
) -> str:
    """Build a prompt that asks the LLM to select the correct function name.

    Args:
        prompt: The original user natural language prompt.
        functions: List of available function definitions.

    Returns:
        A formatted system+user prompt string.
    """
    fn_descriptions = "\n".join(
        f"- {fn.name}: {fn.description}" for fn in functions
    )
    return (
        "You are a function-calling assistant. "
        "Given a user request, you must select the most appropriate function.\n\n"
        f"Available functions:\n{fn_descriptions}\n\n"
        f"User request: {prompt}\n\n"
        "Respond with ONLY the function name, nothing else.\n"
        "Function name: "
    )


def _build_argument_extraction_prompt(
    prompt: str,
    function: FunctionDefinition,
) -> str:
    """Build a prompt that asks the LLM to extract arguments as JSON.

    Args:
        prompt: The original user natural language prompt.
        function: The selected function definition.

    Returns:
        A formatted prompt string asking for JSON arguments.
    """
    params_desc = ", ".join(
        f"{name} ({info.type})" for name, info in function.parameters.items()
    )
    return (
        "You are a function-calling assistant. "
        "Extract the arguments for the given function from the user request.\n\n"
        f"Function: {function.name}\n"
        f"Description: {function.description}\n"
        f"Parameters: {params_desc}\n\n"
        f"User request: {prompt}\n\n"
        "Respond with ONLY a valid JSON object containing the arguments.\n"
        "Arguments: "
    )


class FunctionCaller:
    """Orchestrates LLM-based function selection and argument extraction.

    Args:
        model: The Small_LLM_Model SDK instance.
        vocab: The loaded Vocabulary instance.
        max_new_tokens: Maximum tokens to generate per call.
    """

    def __init__(
        self,
        model: Any,
        vocab: Vocabulary,
        max_new_tokens: int = 256,
    ) -> None:
        """Initialize the FunctionCaller.

        Args:
            model: LLM model instance from llm_sdk.
            vocab: Vocabulary instance.
            max_new_tokens: Token generation limit.
        """
        self.model = model
        self.vocab = vocab
        self.decoder = ConstrainedDecoder(vocab, max_new_tokens=max_new_tokens)

    def process_prompt(
        self,
        prompt: Prompt,
        functions: List[FunctionDefinition],
    ) -> FunctionCall:
        """Process a single prompt and produce a FunctionCall result.

        Pass 1: Use LLM + constrained decoding to select the function name.
        Pass 2: Use LLM + constrained decoding to extract the arguments.

        Args:
            prompt: The input Prompt object.
            functions: List of available FunctionDefinition objects.

        Returns:
            A FunctionCall with prompt, name, and parameters.
        """
        function_names = [fn.name for fn in functions]

        # ---- Pass 1: Select function name ----
        selection_text = _build_function_selection_prompt(
            prompt.prompt, functions
        )
        input_ids_1: List[int] = self.model.encode(selection_text).squeeze(0).tolist()
        chosen_name = self.decoder.generate_function_name(
            self.model, input_ids_1, function_names
        )

        # Find the selected function definition
        selected_fn: FunctionDefinition = functions[0]
        for fn in functions:
            if fn.name == chosen_name:
                selected_fn = fn
                break

        # ---- Pass 2: Extract arguments ----
        schema: Dict[str, str] = {
            name: ptype.type
            for name, ptype in selected_fn.parameters.items()
        }

        parameters: Dict[str, Any]
        if not schema:
            parameters = {}
        else:
            arg_text = _build_argument_extraction_prompt(prompt.prompt, selected_fn)
            input_ids_2: List[int] = self.model.encode(arg_text).squeeze(0).tolist()
            parameters = self.decoder.generate_parameters(
                self.model, input_ids_2, schema
            )

        return FunctionCall(
            prompt=prompt.prompt,
            name=chosen_name,
            parameters=parameters,
        )

    def process_all(
        self,
        prompts: List[Prompt],
        functions: List[FunctionDefinition],
    ) -> List[FunctionCall]:
        """Process all prompts and return a list of FunctionCall results.

        Args:
            prompts: List of input Prompt objects.
            functions: List of available FunctionDefinition objects.

        Returns:
            List of FunctionCall results.
        """
        results: List[FunctionCall] = []
        for i, prompt in enumerate(prompts):
            print(f"  [{i + 1}/{len(prompts)}] Processing: '{prompt.prompt[:60]}...'")
            try:
                result = self.process_prompt(prompt, functions)
                results.append(result)
                print(f"    -> {result.name}({result.parameters})")
            except Exception as e:
                print(f"    [ERROR] Failed to process prompt: {e}")
                # Graceful fallback: use first function with empty params
                results.append(FunctionCall(
                    prompt=prompt.prompt,
                    name=functions[0].name,
                    parameters={},
                ))
        return results
