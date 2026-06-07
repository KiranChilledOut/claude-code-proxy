from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ResponsesItem(BaseModel):
    type: str
    id: Optional[str] = None
    role: Optional[str] = None
    status: Optional[str] = None
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    call_id: Optional[str] = None
    name: Optional[str] = None
    namespace: Optional[str] = None
    arguments: Optional[str] = None
    output: Optional[Any] = None

    class Config:
        extra = "allow"


class ResponsesRequest(BaseModel):
    model: str
    instructions: Optional[str] = None
    input: Union[str, List[ResponsesItem]]
    previous_response_id: Optional[str] = None
    store: Optional[bool] = None
    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False
    tools: Optional[List[Any]] = None
    tool_choice: Optional[Any] = None
    parallel_tool_calls: Optional[bool] = None
    reasoning: Optional[Dict[str, Any]] = None
    user: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"


class InputTokensDetails(BaseModel):
    cached_tokens: Optional[int] = None

    class Config:
        extra = "allow"


class OutputTokensDetails(BaseModel):
    reasoning_tokens: Optional[int] = None

    class Config:
        extra = "allow"


class ResponsesUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: Optional[InputTokensDetails] = None
    output_tokens_details: Optional[OutputTokensDetails] = None
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None

    class Config:
        extra = "allow"


class ResponsesResponse(BaseModel):
    id: str
    model: str
    output: List[ResponsesItem]
    status: str
    previous_id: Optional[str] = None
    usage: ResponsesUsage

    class Config:
        extra = "allow"


class ResponsesStreamEvent(BaseModel):
    type: str
    id: Optional[str] = None
    model: Optional[str] = None
    output_index: Optional[int] = None
    item: Optional[ResponsesItem] = None
    delta: Optional[str] = None
    usage: Optional[ResponsesUsage] = None

    class Config:
        extra = "allow"
