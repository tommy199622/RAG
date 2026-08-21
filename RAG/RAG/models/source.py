from typing import Optional

from pydantic import BaseModel



class Source(BaseModel):

    filename: str

    page: int

    filepath: Optional[str] = None

    category: Optional[str] = None