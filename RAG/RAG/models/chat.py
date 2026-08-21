from pydantic import BaseModel,Field
from typing import List, Optional

from models.source import Source


class ChatRequest(BaseModel):
    session_id: str = Field(
        ...,
        description="對話 Session ID"
    )
    question: str = Field(
        ...,
        min_length=1,
        description="使用者問題"
    )

class ChatResponse(BaseModel):

    answer: str

    sources: List[Source] = []
    current_model: Optional[str] = None