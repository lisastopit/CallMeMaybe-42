"""Constrained decoding engine for guaranteed valid JSON generation."""

import json
import math
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .vocabulary import Vocabulary


class JsonState(Enum):
    START = auto()
    KEY = auto()
    COLON = auto()
    VALUE_START = auto()
    VALUE_STRING = auto()
    VALUE_NUMBER = auto()
    VALUE_BOOL_TRUE = auto()
    VALUE_BOOL_FALSE = auto()
    COMMA_OR_END = auto()
    DONE = auto()


class ConstrainedDecoder:
    NEG_INF: float = -math.inf

    def __init__(self, vocab: Vocabulary, max_new_tokens: int = 256) -> None:
        self.vocab = vocab
        self.max_new_tokens = max_new_tokens

    def generate_function_name(
        self,
        model: Any,
        input_ids: List[int],
        function_names: List[str],
    ) -> str:
        current_ids = list(input_ids)
        generated_text = ""

        for _ in range(self.max_new_tokens):
            raw_logits = model.get_logits_from_input_ids(current_ids)
            next_logits: np.ndarray = np.array(raw_logits, dtype=np.float32)

            valid_token_ids = self._get_valid_name_tokens(
                generated_text, function_names
            )

            if not valid_token_ids:
                break

            if generated_text in function_names:
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

            clean = chosen_token.lstrip("\u0120").lstrip("Ġ")
            generated_text += clean
            current_ids.append(chosen_id)

            if generated_text in function_names:
                longer_matches = [
                    n for n in function_names
                    if n.startswith(generated_text) and n != generated_text
                ]
                if not longer_matches:
                    break

        if generated_text not in function_names:
            generated_text = self._best_match(generated_text, function_names)

        return generated_text

    def generate_parameters(
        self,
        model: Any,
        input_ids: List[int],
        schema: Dict[str, str],
    ) -> Dict[str, Any]:
        current_ids = list(input_ids)
        generated_chars = ""
        state = JsonState.START
        current_key = ""
        current_value_raw = ""
        result: Dict[str, Any] = {}
        remaining_keys = list(schema.keys())

        string_depth = 0
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

            char = chosen_token.lstrip("\u0120").lstrip("Ġ")

            state, current_key, current_value_raw, remaining_keys, result, \
                string_depth, string_escape_next = self._advance_state(
                    state, char, current_key, current_value_raw,
                    remaining_keys, result, schema, string_depth, string_escape_next
                )

            generated_chars += char
            current_ids.append(chosen_id)

            if state == JsonState.DONE:
                break

        return self._safe_parse(generated_chars, schema, result)

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
        if state == JsonState.START:
            if char == "{":
                state = JsonState.KEY if remaining_keys else JsonState.COMMA_OR_END

        elif state == JsonState.KEY:
            current_key += char
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
                    key = current_key.strip('"')
                    try:
                        parsed_value = json.loads(current_value_raw)
                        result[key] = parsed_value
                    except json.JSONDecodeError:
                        result[key] = current_value_raw.strip('"')
                    
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

    def _get_valid_name_tokens(
        self, generated_so_far: str, function_names: List[str]
    ) -> List[int]:
        candidates = [n for n in function_names if n.startswith(generated_so_far)]
        if not candidates:
            return []

        valid_ids: List[int] = []
        for token, tid in self.vocab.token_to_id.items():
            clean = token.lstrip("\u0120").lstrip("Ġ")
            if not clean:
                continue
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
        allowed_chars: Optional[List[str]] = None

        if state == JsonState.START:
            allowed_chars = ["{"]

        elif state == JsonState.KEY:
            if not current_key:
                allowed_chars = ['"']
            else:
                partial = current_key[1:]
                matching = [k for k in remaining_keys if k.startswith(partial)]
                if not matching:
                    allowed_chars = []
                else:
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
            allowed_chars = None

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

        return self._chars_to_token_ids(allowed_chars, state)

    def _chars_to_token_ids(
        self,
        allowed_chars: Optional[List[str]],
        state: JsonState,
    ) -> List[int]:
        valid: List[int] = []

        if state == JsonState.VALUE_STRING and allowed_chars is None:
            # Allow any printable character including space
            for token, tid in self.vocab.token_to_id.items():
                clean = token.lstrip("\u0120").lstrip("Ġ")
                if len(clean) == 1 and (clean.isprintable() or clean == " "):
                    valid.append(tid)
            return valid

        if allowed_chars is None:
            return list(self.vocab.token_to_id.values())

        allowed_set = set(allowed_chars)
        for token, tid in self.vocab.token_to_id.items():
            clean = token.lstrip("\u0120").lstrip("Ġ")
            if clean and clean in allowed_set:
                valid.append(tid)

        return valid

    def _mask_logits(
        self, logits: np.ndarray, valid_ids: List[int]
    ) -> np.ndarray:
        masked = np.full_like(logits, self.NEG_INF)
        for vid in valid_ids:
            if 0 <= vid < len(masked):
                masked[vid] = logits[vid]
        return masked

    def _best_match(self, generated: str, function_names: List[str]) -> str:
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
        try:
            parsed = json.loads(generated)
            if isinstance(parsed, dict):
                return self._coerce_types(parsed, schema)
        except (json.JSONDecodeError, ValueError):
            pass

        for suffix in ["}", "}}"]:
            try:
                parsed = json.loads(generated + suffix)
                if isinstance(parsed, dict):
                    return self._coerce_types(parsed, schema)
            except (json.JSONDecodeError, ValueError):
                pass

        return self._coerce_types(partial, schema)

    def _coerce_types(
        self, data: Dict[str, Any], schema: Dict[str, str]
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, expected_type in schema.items():
            if key not in data:
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
