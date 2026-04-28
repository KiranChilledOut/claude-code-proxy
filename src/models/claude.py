from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class ClaudeContentBlockText(BaseModel):
    type: Literal["text"]
    text: str


class ClaudeContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]


class ClaudeContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]


class ClaudeContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any]]


class ClaudeSystemContent(BaseModel):
    type: Literal["text"]
    text: str


class ClaudeMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[
        str,
        List[
            Union[
                ClaudeContentBlockText,
                ClaudeContentBlockImage,
                ClaudeContentBlockToolUse,
                ClaudeContentBlockToolResult,
            ]
        ],
    ]


class ClaudeTool(BaseModel):
    """Represents both standard function tools and schema-less Anthropic tools.

    Standard tools have: name, input_schema, optional description
    Schema-less tools have: type (e.g. "computer_20251124"), name, and
    type-specific fields (display_width_px, display_height_px, etc.)
    """

    # Common fields
    name: str
    type: Optional[str] = None  # e.g. "computer_20251124", "bash_20250124", "text_editor_20250728"
    description: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None  # None for schema-less tools

    # Computer use specific fields
    display_width_px: Optional[int] = None
    display_height_px: Optional[int] = None
    display_number: Optional[int] = None
    enable_zoom: Optional[bool] = None

    class Config:
        extra = "allow"  # Allow extra fields for forward compatibility

    def is_schema_less(self) -> bool:
        """Check if this is a schema-less Anthropic tool (computer, bash, text_editor)."""
        if self.type is None:
            return False
        return any(
            self.type.startswith(prefix) for prefix in ("computer_", "bash_", "text_editor_")
        )


class ClaudeThinkingConfig(BaseModel):
    enabled: bool = True


class ClaudeMessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[ClaudeMessage]
    system: Optional[Union[str, List[ClaudeSystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[ClaudeTool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ClaudeThinkingConfig] = None
    # Beta headers forwarded by Anthropic SDK clients
    betas: Optional[List[str]] = None

    class Config:
        extra = "allow"  # Forward compatibility


class ClaudeTokenCountRequest(BaseModel):
    model: str
    messages: List[ClaudeMessage]
    system: Optional[Union[str, List[ClaudeSystemContent]]] = None
    tools: Optional[List[ClaudeTool]] = None
    thinking: Optional[ClaudeThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None
