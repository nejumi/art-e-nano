from typing import List, Optional

from pydantic import BaseModel, Field


class SyntheticQuery(BaseModel):
    id: int
    question: str
    answer: str
    message_ids: List[str]
    how_realistic: float
    inbox_address: str
    query_date: str


class Email(BaseModel):
    message_id: str
    date: str
    subject: Optional[str] = None
    from_address: Optional[str] = None
    to_addresses: List[str] = Field(default_factory=list)
    cc_addresses: List[str] = Field(default_factory=list)
    bcc_addresses: List[str] = Field(default_factory=list)
    body: Optional[str] = None
    file_name: Optional[str] = None
