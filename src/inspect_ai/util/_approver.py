from enum import Enum, auto
from typing import Any, Callable
from inspect_ai.tool._tool_call import ToolCall

class ApprovalResult(Enum):
    APPROVE = auto()
    REJECT = auto()
    BLOCK = auto()
    ESCALATE = auto()

ApprovalFunction = Callable[[ToolCall, Any], ApprovalResult]