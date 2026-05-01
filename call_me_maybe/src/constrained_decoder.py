"""Constrained decoding engine for guaranteed valid JSON generation.

This module implements token-by-token constrained generation that enforces
both syntactic JSON validity and semantic schema compliance at every step.
The decoder tracks a Finite State Machine (FSM) over JSON structure states
and, at each generation step, masks out any tokens that would violate the
current state or the required schema.
"""

import json
import math
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .vocabulary import Vocabulary

# ---------------------------------------------------------------------------
# JSON generation state machine
# ---------------------------------------------------------------------------


class JsonState(Enum):
    """States of the JSON object generation FSM."""
    START = auto()             # Expecting opening '{'
    KEY = auto()               # Expecting a key string
    COLON = auto()             # Expecting ':'
    VALUE_START = auto()       # Expecting the start of a value
    VALUE_STRING = auto()      # Inside a string value (character by character)
    VALUE_NUMBER = auto()      # Inside a number value
    VALUE_BOOL_TRUE = auto()   # Writing 'true'
    VALUE_BOOL_FALSE = auto()  # Writing 'false'
    COMMA_OR_END = auto()      # Expecting ',' or '}'
    DONE = auto()              # Generation complete


class ConstrainedDecoder:
    """Generates structured JSON constrained to a given parameter schema.

    This decoder is used in two passes:
      1. Function selection: generate a single string token from the list of
         available function names.
      2. Parameter generation: generate a JSON object conforming to the
         function's parameter schema.

    Args:
        vocab: The loaded vocabulary object.
        max_new_tokens: Maximum number of tokens to generate before aborting.
    """

    NEG_INF: float = -math.inf

    def __init__(self, vocab: Vocabulary, max_new_tokens: int = 512) -> None:
        """Initialize the constrained decoder.

        Args:
            vocab: Vocabulary instance loaded from the SDK.
            max_new_tokens: Safety cap on generation length.
        """
        self.vocab = vocab
        self.max_new_tokens = max_new_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_function_name(
        self,
        model: Any,
        input_ids: List[int],
        function_names: List[str],
    ) -> str:
        """Generate a function name constrained to the allowed list.

        Uses the LLM logits but restricts valid tokens to only those that
        form one of the provided function names, token by token.

        Args:
            model: The Small_LLM_Model instance.
            input_ids: Tokenized prompt.
            function_names: List of valid function name strings.

        Returns:
            The selected function name string.
        """

        current_ids = list(input_ids)
        generated_text = ""

        for _ in range(self.max_new_tokens):

            raw_logits = model.get_logits_from_input_ids(current_ids)

            next_logits: np.ndarray = np.array(raw_logits, dtype=np.float32)

            # Find valid tokens: those that extend a still-matching function name
            valid_token_ids = self._get_valid_name_tokens(
                generated_text, function_names
            )

            if not valid_token_ids:
                # Should not happen if function_names is non-empty
                break

            # Check if we've already matched a complete function name
            if generated_text in function_names:
                # Check if any function name is a prefix of another
                longer_matches = [
                    n for n in function_names
                    if n.startswith(generated_text) and n != generated_text
                ]
                if not longer_matches:
                    break

            masked = self._mask_logits(next_logits, valid_token_ids)
            chosen_id = int(np.argmax(masked))
            chosen_token = self.vocab.get_token(chosen_id)
            if chosen_token is None:
                break

            # Strip leading-space marker (Ġ / \u0120 used by some tokenizers)
            clean = chosen_token.lstrip("\u0120").lstrip("Ġ")
            generated_text += clean
            current_ids.append(chosen_id)

            if generated_text in function_names:
                # Verify no longer prefix match exists
                longer_matches = [
                    n for n in function_names
                    if n.startswith(generated_text) and n != generated_text
                ]
                if not longer_matches:
                    break

        # Fallback: if result not clean, find best matching function name
        if generated_text not in function_names:
            generated_text = self._best_match(generated_text, function_names)

        return generated_text

    def generate_parameters(
        self,
        model: Any,
        input_ids: List[int],
        schema: Dict[str, str],
    ) -> Dict[str, Any]:
        """Generate a JSON object constrained to a given parameter schema.

        Args:
            model: The Small_LLM_Model instance.
            input_ids: Tokenized prompt (including function name context).
            schema: Dict mapping parameter name -> type string.

        Returns:
            Dict of parameter name -> parsed value.
        """

        current_ids = list(input_ids)
        generated_chars = ""
        state = JsonState.START
        current_key = ""
        current_value_raw = ""
        result: Dict[str, Any] = {}
        remaining_keys = list(schema.keys())

        string_depth = 0  # tracks if we're inside a string value
        string_escape_next = False

        for _ in range(self.max_new_tokens):

            raw_logits = model.get_logits_from_input_ids(current_ids)
            next_logits: np.ndarray = np.array(raw_logits, dtype=np.float32)

            valid_ids = self._get_valid_json_tokens(
                state, generated_chars, current_key, current_value_raw,
                schema, remaining_keys, string_depth, string_escape_next
            )

            if not valid_ids:
                break

            masked = self._mask_logits(next_logits, valid_ids)
            chosen_id = int(np.argmax(masked))
            chosen_token = self.vocab.get_token(chosen_id)
            if chosen_token is None:
                break

            # Decode the token to a character string (strip tokenizer space prefix)
            char = chosen_token.lstrip("\u0120").lstrip("Ġ")

            # Update FSM state
            state, current_key, current_value_raw, remaining_keys, result, \
                string_depth, string_escape_next = self._advance_state(
                    state, char, current_key, current_value_raw,
                    remaining_keys, result, schema, string_depth, string_escape_next
                )

            generated_chars += char
            current_ids.append(chosen_id)

            if state == JsonState.DONE:
                break

        # Try to parse what we have so far
        return self._safe_parse(generated_chars, schema, result)

    # ------------------------------------------------------------------
    # FSM helpers
    # ------------------------------------------------------------------

    def _advance_state(
        self,
        state: JsonState,
        char: str,
        current_key: str,
        current_value_raw: str,
        remaining_keys: List[str],
        result: Dict[str, Any],
        schema: Dict[str, str],
        string_depth: int,
        escape_next: bool,
    ) -> Tuple[
        JsonState, str, str, List[str], Dict[str, Any], int, bool
    ]:
        """Advance the FSM by one character.

        Args:
            state: Current FSM state.
            char: The character(s) being appended.
            current_key: Key being built or currently active.
            current_value_raw: Raw accumulated value string.
            remaining_keys: Keys not yet generated.
            result: Accumulated result dict.
            schema: Parameter schema.
            string_depth: Nesting depth inside string (0 = not in string).
            escape_next: Whether next char is escaped.

        Returns:
            Updated tuple of all mutable state.
        """
        if state == JsonState.START:
            if char == "{":
                state = JsonState.KEY if remaining_keys else JsonState.COMMA_OR_END

        elif state == JsonState.KEY:
            # We're building the key; it starts with " and ends with "
            current_key += char
            # Remove surrounding quotes when complete
            if current_key.startswith('"') and current_key.endswith('"') and len(current_key) > 1:
                state = JsonState.COLON

        elif state == JsonState.COLON:
            if char == ":":
                current_value_raw = ""
                state = JsonState.VALUE_START

        elif state == JsonState.VALUE_START:
            if char == '"':
                current_value_raw = '"'
                string_depth = 1
                state = JsonState.VALUE_STRING
            elif char in "-0123456789":
                current_value_raw = char
                state = JsonState.VALUE_NUMBER
            elif char == "t":
                current_value_raw = "t"
                state = JsonState.VALUE_BOOL_TRUE
            elif char == "f":
                current_value_raw = "f"
                state = JsonState.VALUE_BOOL_FALSE

        elif state == JsonState.VALUE_STRING:
            if escape_next:
                current_value_raw += char
                escape_next = False
            elif char == "\\":
                current_value_raw += char
                escape_next = True
            elif char == '"':
                current_value_raw += '"'
                string_depth -= 1
                if string_depth == 0:
                    # String complete — store value
                    key = current_key.strip('"')
                    result[key] = json.loads(current_value_raw)
                    if key in remaining_keys:
                        remaining_keys = [k for k in remaining_keys if k != key]
                    current_key = ""
                    current_value_raw = ""
                    state = JsonState.COMMA_OR_END
            else:
                current_value_raw += char

        elif state == JsonState.VALUE_NUMBER:
            if char in "0123456789.eE+-":
                current_value_raw += char
            else:
                # Number ended
                key = current_key.strip('"')
                try:
                    num = json.loads(current_value_raw)
                    result[key] = float(num) if schema.get(key) == "number" else num
                except (ValueError, json.JSONDecodeError):
                    result[key] = 0.0
                if key in remaining_keys:
                    remaining_keys = [k for k in remaining_keys if k != key]
                current_key = ""
                current_value_raw = ""
                if char == ",":
                    state = JsonState.KEY
                elif char == "}":
                    state = JsonState.DONE

        elif state == JsonState.VALUE_BOOL_TRUE:
            current_value_raw += char
            if current_value_raw == "true":
                key = current_key.strip('"')
                result[key] = True
                if key in remaining_keys:
                    remaining_keys = [k for k in remaining_keys if k != key]
                current_key = ""
                current_value_raw = ""
                state = JsonState.COMMA_OR_END

        elif state == JsonState.VALUE_BOOL_FALSE:
            current_value_raw += char
            if current_value_raw == "false":
                key = current_key.strip('"')
                result[key] = False
                if key in remaining_keys:
                    remaining_keys = [k for k in remaining_keys if k != key]
                current_key = ""
                current_value_raw = ""
                state = JsonState.COMMA_OR_END

        elif state == JsonState.COMMA_OR_END:
            if char == "," and remaining_keys:
                current_key = ""
                state = JsonState.KEY
            elif char == "}":
                state = JsonState.DONE

        return (
            state, current_key, current_value_raw,
            remaining_keys, result, string_depth, escape_next
        )

    # ------------------------------------------------------------------
    # Token masking helpers
    # ------------------------------------------------------------------

    def _get_valid_name_tokens(
        self, generated_so_far: str, function_names: List[str]
    ) -> List[int]:
        """Return token IDs that validly extend the function name being generated.

        Args:
            generated_so_far: Characters generated so far.
            function_names: All valid function name options.

        Returns:
            List of valid token IDs.
        """
        # Find names that still start with what we've generated
        candidates = [n for n in function_names if n.startswith(generated_so_far)]
        if not candidates:
            return []

        valid_ids: List[int] = []
        for token, tid in self.vocab.token_to_id.items():
            # Clean the token (remove leading space marker)
            clean = token.lstrip("\u0120").lstrip("Ġ")
            if not clean:
                continue
            # This token is valid if appending it still matches a candidate
            proposed = generated_so_far + clean
            for cand in candidates:
                if cand.startswith(proposed) or proposed == cand:
                    valid_ids.append(tid)
                    break
        return valid_ids

    def _get_valid_json_tokens(
        self,
        state: JsonState,
        generated: str,
        current_key: str,
        current_value_raw: str,
        schema: Dict[str, str],
        remaining_keys: List[str],
        string_depth: int,
        escape_next: bool,
    ) -> List[int]:
        """Return token IDs valid in the current JSON FSM state.

        Args:
            state: Current FSM state.
            generated: Full generated string so far.
            current_key: Currently active key.
            current_value_raw: Raw value accumulated so far.
            schema: Parameter schema.
            remaining_keys: Keys not yet generated.
            string_depth: String nesting depth.
            escape_next: Whether the next char is escaped.

        Returns:
            List of valid token IDs.
        """
        allowed_chars: Optional[List[str]] = None

        if state == JsonState.START:
            allowed_chars = ["{"]

        elif state == JsonState.KEY:
            # We're building a key from remaining_keys
            # Key must be one of remaining_keys surrounded by quotes
            if not current_key:
                allowed_chars = ['"']
            else:
                # current_key starts with '"', find what keys still match
                partial = current_key[1:]  # strip opening quote
                matching = [k for k in remaining_keys if k.startswith(partial)]
                if not matching:
                    allowed_chars = []
                else:
                    # Chars that can extend the partial key, or closing " if exact match
                    next_chars: List[str] = []
                    for k in matching:
                        if k == partial:
                            next_chars.append('"')
                        elif len(k) > len(partial):
                            next_chars.append(k[len(partial)])
                    allowed_chars = list(set(next_chars))

        elif state == JsonState.COLON:
            allowed_chars = [":"]

        elif state == JsonState.VALUE_START:
            param_type = schema.get(current_key.strip('"'), "string")
            if param_type == "string":
                allowed_chars = ['"']
            elif param_type in ("number", "integer"):
                allowed_chars = ["-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            elif param_type == "boolean":
                allowed_chars = ["t", "f"]
            else:
                allowed_chars = ['"']

        elif state == JsonState.VALUE_STRING:
            if escape_next:
                allowed_chars = None  # any printable char
            else:
                # Any character is valid in a string except unescaped control chars
                # We'll allow all single-char tokens that don't break JSON
                allowed_chars = None  # handled below as "any non-control"

        elif state == JsonState.VALUE_NUMBER:
            param_type = schema.get(current_key.strip('"'), "number")
            if param_type == "integer":
                allowed_chars = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ",", "}"]
            else:
                allowed_chars = [
                    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                    ".", "e", "E", "+", "-", ",", "}"
                ]

        elif state == JsonState.VALUE_BOOL_TRUE:
            remaining = "true"[len(current_value_raw):]
            allowed_chars = [remaining[0]] if remaining else [",", "}"]

        elif state == JsonState.VALUE_BOOL_FALSE:
            remaining = "false"[len(current_value_raw):]
            allowed_chars = [remaining[0]] if remaining else [",", "}"]

        elif state == JsonState.COMMA_OR_END:
            if remaining_keys:
                allowed_chars = [","]
            else:
                allowed_chars = ["}"]

        elif state == JsonState.DONE:
            return []

        # Map allowed_chars to token IDs
        return self._chars_to_token_ids(allowed_chars, state)

    def _chars_to_token_ids(
        self,
        allowed_chars: Optional[List[str]],
        state: JsonState,
    ) -> List[int]:
        """Convert allowed character list to token IDs.

        Args:
            allowed_chars: List of allowed single characters, or None for any.
            state: Current FSM state (used for string-interior handling).

        Returns:
            List of valid token IDs.
        """
        valid: List[int] = []

        if state == JsonState.VALUE_STRING and allowed_chars is None:
            # Inside string: allow any single-char token that is printable
            # plus backslash and closing quote
            for token, tid in self.vocab.token_to_id.items():
                clean = token.lstrip("\u0120").lstrip("Ġ")
                if len(clean) == 1 and (clean.isprintable() or clean in "\t\n"):
                    valid.append(tid)
            return valid

        if allowed_chars is None:
            return list(self.vocab.token_to_id.values())

        allowed_set = set(allowed_chars)
        for token, tid in self.vocab.token_to_id.items():
            clean = token.lstrip("\u0120").lstrip("Ġ")
            if clean and clean in allowed_set:
                valid.append(tid)
            elif clean and len(clean) > 1:
                # Multi-char token: only allow if entire token matches one allowed string
                if clean in allowed_set:
                    valid.append(tid)

        return valid

    def _mask_logits(
        self, logits: np.ndarray, valid_ids: List[int]
    ) -> np.ndarray:
        """Set all logits except valid_ids to -inf.

        Args:
            logits: Raw logit array of shape (vocab_size,).
            valid_ids: List of valid token IDs.

        Returns:
            Masked logit array.
        """
        masked = np.full_like(logits, self.NEG_INF)
        for vid in valid_ids:
            if 0 <= vid < len(masked):
                masked[vid] = logits[vid]
        return masked

    def _best_match(self, generated: str, function_names: List[str]) -> str:
        """Find the function name with the longest common prefix match.

        Args:
            generated: The partially generated string.
            function_names: All valid function names.

        Returns:
            Best matching function name.
        """
        best = function_names[0]
        best_len = 0
        for name in function_names:
            common = 0
            for a, b in zip(generated, name):
                if a == b:
                    common += 1
                else:
                    break
            if common > best_len:
                best_len = common
                best = name
        return best

    def _safe_parse(
        self,
        generated: str,
        schema: Dict[str, str],
        partial: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Attempt to parse the generated JSON, falling back to partial result.

        Args:
            generated: The full generated JSON string.
            schema: Parameter schema for type coercion.
            partial: Partially accumulated result.

        Returns:
            Parsed and type-coerced parameter dict.
        """
        # Try full parse
        try:
            parsed = json.loads(generated)
            if isinstance(parsed, dict):
                return self._coerce_types(parsed, schema)
        except (json.JSONDecodeError, ValueError):
            pass

        # Try closing any open braces
        for suffix in ["}", "}}"]:
            try:
                parsed = json.loads(generated + suffix)
                if isinstance(parsed, dict):
                    return self._coerce_types(parsed, schema)
            except (json.JSONDecodeError, ValueError):
                pass

        # Fall back to partial result
        return self._coerce_types(partial, schema)

    def _coerce_types(
        self, data: Dict[str, Any], schema: Dict[str, str]
    ) -> Dict[str, Any]:
        """Coerce parsed values to the types specified in the schema.

        Args:
            data: Parsed parameter dict.
            schema: Expected types per parameter.

        Returns:
            Type-coerced dict.
        """
        result: Dict[str, Any] = {}
        for key, expected_type in schema.items():
            if key not in data:
                # Provide a default value for missing keys
                if expected_type == "number":
                    result[key] = 0.0
                elif expected_type == "integer":
                    result[key] = 0
                elif expected_type == "boolean":
                    result[key] = False
                else:
                    result[key] = ""
                continue

            value = data[key]
            try:
                if expected_type == "number":
                    result[key] = float(value)
                elif expected_type == "integer":
                    result[key] = int(value)
                elif expected_type == "boolean":
                    result[key] = bool(value)
                elif expected_type == "string":
                    result[key] = str(value)
                else:
                    result[key] = value
            except (ValueError, TypeError):
                result[key] = value

        return result
