# myproject/models.py
from pydantic import BaseModel
from typing import Dict, Any, Literal

class TextProblemStatement(BaseModel):
    """
    文本类问题描述。
    """

    text: str
    extra_fields: Dict[str, Any] = {}
    type: Literal["text"] = "text"
    id: str | None = None

    def get_problem_statement(self) -> str:
        """返回问题文本。"""
        return self.text
