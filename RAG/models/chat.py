from pydantic import BaseModel
from typing import List, Optional

from models.source import Source


class ChatRequest(BaseModel):

    # session_id: Optional[str] = None

    question: str


class ChatResponse(BaseModel):

    answer: str

    sources: List[Source]