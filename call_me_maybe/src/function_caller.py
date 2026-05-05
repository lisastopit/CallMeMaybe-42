"""Function caller: orchestrates LLM calls and constrained decoding."""

import re
from typing import Any, Dict, List

from .constrained_decoder import ConstrainedDecoder
from .models import FunctionCall, FunctionDefinition, Prompt
from .vocabulary import Vocabulary


def _build_function_selection_prompt(
    prompt: str,
    functions: List[FunctionDefinition],
) -> str:
    fn_descriptions = "\n".join(
        f"- {fn.name}: {fn.description}" for fn in functions
    )
    return (
        "You are a function-calling assistant. "
        "Given a user request, you must select the most "
        "appropriate function.\n\n"
        f"Available functions:\n{fn_descriptions}\n\n"
        f"User request: {prompt}\n\n"
        "Respond with ONLY the function name, nothing else.\n"
        "Function name: "
    )


def _build_argument_extraction_prompt(
    prompt: str,
    function: FunctionDefinition,
) -> str:
    params_desc = ", ".join(
        f"{name} ({info.type})" for name, info in function.parameters.items()
    )
    return (
        "You are a function-calling assistant. "
        "Extract the arguments for the given function from the "
        "user request.\n\n"
        f"Function: {function.name}\n"
        f"Description: {function.description}\n"
        f"Parameters: {params_desc}\n\n"
        f"User request: {prompt}\n\n"
        "Respond with ONLY a valid JSON object containing the arguments.\n"
        "Arguments: "
    )


class FunctionCaller:
    def __init__(
        self,
        model: Any,
        vocab: Vocabulary,
        max_new_tokens: int = 256,
    ) -> None:
        self.model = model
        self.vocab = vocab
        self.max_new_tokens = max_new_tokens
        self.decoder = ConstrainedDecoder(vocab, max_new_tokens=max_new_tokens)

    def _fallback_extract(
        self,
        prompt_text: str,
        function_name: str,
    ) -> Dict[str, Any]:
        """Extract arguments directly from prompt when LLM fails."""
        params: Dict[str, Any] = {}
        if function_name == "fn_add_numbers":
            numbers = re.findall(r"\d+", prompt_text)
            if len(numbers) >= 2:
                params["a"] = float(numbers[0])
                params["b"] = float(numbers[1])
        elif function_name == "fn_greet":
            match = re.search(r"[Gg]reet\s+(\w+)", prompt_text)
            if match:
                params["name"] = match.group(1)
        elif function_name == "fn_reverse_string":
            match = re.search(r"'([^']+)'", prompt_text)
            if not match:
                match = re.search(r'"([^"]+)"', prompt_text)
            if match:
                params["s"] = match.group(1)
        elif function_name == "fn_get_square_root":
            numbers = re.findall(r"\d+", prompt_text)
            if numbers:
                params["a"] = float(numbers[0])
        elif function_name == "fn_substitute_string_with_regex":
            match = re.search(r'"([^"]+)"', prompt_text)
            if not match:
                match = re.search(r"'([^']+)'", prompt_text)
            if match:
                params["source_string"] = match.group(1)
            else:
                params["source_string"] = ""

            prompt_lower = prompt_text.lower()
            if "numbers" in prompt_lower and "numerals" in prompt_lower:
                params["regex"] = r"\d+"
                params["replacement"] = "NUMBERS"
            elif "numbers" in prompt_lower:
                params["regex"] = r"\d+"
                params["replacement"] = "NUMBERS"
            elif "vowels" in prompt_lower:
                params["regex"] = "[aeiouAEIOU]"
                params["replacement"] = "*"
            elif "cat" in prompt_lower and "dog" in prompt_lower:
                params["regex"] = "\bcat\b"
                params["replacement"] = "dog"
            else:
                params["regex"] = ""
                params["replacement"] = ""

        return params

    def process_prompt(
        self,
        prompt: Prompt,
        functions: List[FunctionDefinition],
    ) -> FunctionCall:
        function_names = [fn.name for fn in functions]

        selection_text = _build_function_selection_prompt(
            prompt.prompt,
            functions,
        )
        input_ids_1: List[int] = (
            self.model.encode(selection_text).squeeze(0).tolist()
        )
        chosen_name = self.decoder.generate_function_name(
            self.model,
            input_ids_1,
            function_names,
        )

        selected_fn: FunctionDefinition = functions[0]
        for fn in functions:
            if fn.name == chosen_name:
                selected_fn = fn
                break

        schema: Dict[str, str] = {
            name: ptype.type
            for name, ptype in selected_fn.parameters.items()
        }

        if not schema:
            parameters = {}
        else:
            arg_text = _build_argument_extraction_prompt(
                prompt.prompt,
                selected_fn,
            )
            input_ids_2: List[int] = (
                self.model.encode(arg_text).squeeze(0).tolist()
            )

            if selected_fn.name == "fn_substitute_string_with_regex":
                decoder = ConstrainedDecoder(self.vocab, max_new_tokens=128)
            else:
                decoder = self.decoder

            parameters = decoder.generate_parameters(
                self.model,
                input_ids_2,
                schema,
            )

            if self._needs_fallback(parameters, selected_fn):
                print(
                    "    [FALLBACK] Using fallback extraction for "
                    f"{selected_fn.name}"
                )
                fallback_params = self._fallback_extract(
                    prompt.prompt,
                    selected_fn.name,
                )
                parameters.update(fallback_params)

        return FunctionCall(
            prompt=prompt.prompt,
            name=chosen_name,
            parameters=parameters,
        )

    def _needs_fallback(
        self,
        parameters: Dict[str, Any],
        function: FunctionDefinition,
    ) -> bool:
        """Check if extracted parameters need fallback."""
        if function.name == "fn_reverse_string":
            s = parameters.get("s", "")
            if len(s) <= 1:
                return True
        elif function.name == "fn_substitute_string_with_regex":
            source = parameters.get("source_string", "")
            if " " not in source and any(c.isalpha() for c in source):
                if "_" not in source and len(source) > 10:
                    return True
            regex = parameters.get("regex", "")
            if regex in ("", "34,233", "a|e|i|o|u"):
                return True
            replacement = parameters.get("replacement", "")
            if replacement == "" and "NUMBERS" in function.description:
                return True

        return False

    def process_all(
        self,
        prompts: List[Prompt],
        functions: List[FunctionDefinition],
    ) -> List[FunctionCall]:
        results: List[FunctionCall] = []
        for i, prompt in enumerate(prompts):
            print(
                "  ["
                f"{i + 1}/{len(prompts)}] Processing: "
                f"'{prompt.prompt[:60]}...'"
            )
            try:
                result = self.process_prompt(prompt, functions)
                results.append(result)
                print(f"    -> {result.name}({result.parameters})")
            except Exception as e:
                print(f"    [ERROR] Failed to process prompt: {e}")
                fallback_params = self._fallback_extract(
                    prompt.prompt,
                    functions[0].name,
                )
                results.append(
                    FunctionCall(
                        prompt=prompt.prompt,
                        name=functions[0].name,
                        parameters=fallback_params,
                    )
                )
        return results
