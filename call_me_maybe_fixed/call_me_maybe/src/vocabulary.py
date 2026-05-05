"""Vocabulary loading and token utilities for constrained decoding.

Supports both plain vocab.json format ({"token": id}) and
the nested tokenizer.json format used by some HF tokenizers.
"""

import json
from typing import Dict, List, Optional, Tuple


class Vocabulary:
    """Loads and provides lookup utilities for the LLM token vocabulary.

    Attributes:
        token_to_id: Mapping from token string to token ID.
        id_to_token: Mapping from token ID to token string.
    """

    def __init__(self, vocab_path: str) -> None:
        """Initialize the vocabulary from a JSON file.

        Supports both:
        - Plain format: {"token": id, ...}
        - HuggingFace tokenizer.json format with nested "model.vocab"

        Args:
            vocab_path: Path to the vocabulary JSON file provided by the SDK.

        Raises:
            FileNotFoundError: If the vocabulary file does not exist.
            ValueError: If the vocabulary file cannot be parsed.
        """
        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                raw: object = json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Vocabulary file not found: '{vocab_path}'"
            ) from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in vocabulary file: {e}") from e

        vocab_dict: Dict[str, int] = {}

        if isinstance(raw, dict):
            # Check if it's a plain {"token": id} dict
            first_val = next(iter(raw.values()), None) if raw else None
            if isinstance(first_val, int):
                # Plain vocab.json format
                vocab_dict = {k: v for k, v in raw.items() if isinstance(v, int)}
            elif isinstance(raw.get("model"), dict):
                # tokenizer.json with nested model.vocab
                model_section = raw.get("model", {})
                if isinstance(model_section, dict):
                    nested = model_section.get("vocab", {})
                    if isinstance(nested, dict):
                        vocab_dict = {
                            k: v for k, v in nested.items() if isinstance(v, int)
                        }
            elif isinstance(raw.get("vocab"), dict):
                # Some tokenizers have top-level "vocab" key
                nested = raw.get("vocab", {})
                if isinstance(nested, dict):
                    vocab_dict = {
                        k: v for k, v in nested.items() if isinstance(v, int)
                    }
            else:
                # Try treating all int-valued keys as vocab
                vocab_dict = {k: v for k, v in raw.items() if isinstance(v, int)}
        else:
            raise ValueError("Vocabulary file must be a JSON object")

        if not vocab_dict:
            raise ValueError(
                f"No valid token->id mappings found in vocabulary file: '{vocab_path}'"
            )

        self.token_to_id: Dict[str, int] = vocab_dict
        self.id_to_token: Dict[int, str] = {v: k for k, v in vocab_dict.items()}

    def get_id(self, token: str) -> Optional[int]:
        """Return the ID for a token string, or None if not found.

        Args:
            token: The token string.

        Returns:
            Token ID or None.
        """
        return self.token_to_id.get(token)

    def get_token(self, token_id: int) -> Optional[str]:
        """Return the token string for a token ID, or None if not found.

        Args:
            token_id: The token ID.

        Returns:
            Token string or None.
        """
        return self.id_to_token.get(token_id)

    def find_tokens_with_prefix(self, prefix: str) -> List[Tuple[int, str]]:
        """Return all (id, token) pairs where the token starts with the given prefix.

        Args:
            prefix: The string prefix to match.

        Returns:
            List of (token_id, token_string) tuples.
        """
        return [
            (tid, tok)
            for tok, tid in self.token_to_id.items()
            if tok.startswith(prefix)
        ]

    def find_tokens_matching(self, allowed: List[str]) -> List[int]:
        """Return token IDs for the given list of exact token strings.

        Args:
            allowed: List of exact token strings to find.

        Returns:
            List of token IDs that exist in the vocabulary.
        """
        result: List[int] = []
        for token in allowed:
            tid = self.token_to_id.get(token)
            if tid is not None:
                result.append(tid)
        return result
