from typing import Dict, List, Optional, Any
import uuid
from pydantic import BaseModel, Field
# This is a model for aggregating context blocks and model requests/responses and using MCP. 

class ContextBlock(BaseModel):
    query: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


#placeholder for MCP (Model Context Processing) specific fields

class ModelRequest(BaseModel):
    prompt: str
    context: List[ContextBlock] = Field(default_factory=list)
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ModelResponse(BaseModel):
    text: str
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)