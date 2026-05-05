"""Pydantic models for function calling data structures."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, field_validator


class ParameterType(BaseModel):
    """Represents a parameter type definition."""

    type: str

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate that the type is one of the supported types."""
        allowed = {"number", "string", "boolean", "integer", "array", "object"}
        if v not in allowed:
            raise ValueError(f"Unsupported type '{v}'. Must be one of {allowed}")
        return v


class FunctionDefinition(BaseModel):
    """Represents a function definition with name, description, parameters and return type."""

    name: str
    description: str
    parameters: Dict[str, ParameterType]
    returns: Optional[ParameterType] = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate that the function name is not empty."""
        if not v.strip():
            raise ValueError("Function name must not be empty")
        return v.strip()


class Prompt(BaseModel):
    """Represents an input prompt."""

    prompt: str

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """Validate that the prompt is not empty."""
        if not v.strip():
            raise ValueError("Prompt must not be empty")
        return v.strip()


class FunctionCall(BaseModel):
    """Represents the result of a function call: prompt, chosen function and its parameters."""

    prompt: str
    name: str
    parameters: Dict[str, Any]
