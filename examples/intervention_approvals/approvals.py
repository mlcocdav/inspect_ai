from copy import deepcopy
import json
from typing import Any, Protocol, Optional, Literal, cast
from pydantic import BaseModel
import requests
from rich.panel import Panel
from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.prompt import Confirm, Prompt
from inspect_ai.util import input_screen
from inspect_ai.solver._task_state import sample_state, state_jsonable
from inspect_ai.solver import TaskState
from inspect_ai.model import ModelOutput
from inspect_ai.tool import ToolCall
import sys
import shlex
import time
from dataclasses import asdict
from pydantic_core import to_jsonable_python

class ToolCallEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ToolCall):
            return asdict(obj)
        return super().default(obj)

class Approval(BaseModel):
    decision: Literal["approve", "reject", "escalate", "terminate"]
    explanation: str
    modified_tool_call: Optional[ToolCall] = None


class Approver(Protocol):
    """
    Protocol for approvers.
    """

    def __call__(
        self, tool_call: ToolCall, state: Optional[TaskState] = None
    ) -> Approval:
        """
        Approve or reject a tool call.

        Args:
            tool_call (ToolCall): The tool call to be approved.
            state (Optional[TaskState]): The current task state, if available.

        Returns:
            Approval: An Approval object containing the decision and explanation.
        """
        ...


def allow_list_approver(
    allowed_commands: list[str],
    allow_sudo: bool = False,
    command_specific_rules: Optional[dict[str, list[str]]] = None,
) -> Approver:
    """
    Create an approver that checks if a bash command is in an allowed list.

    Args:
        allowed_commands (List[str]): List of allowed bash commands.
        allow_sudo (bool, optional): Whether to allow sudo commands. Defaults to False.
        command_specific_rules (Optional[Dict[str, List[str]]], optional): Dictionary of command-specific rules. Defaults to None.

    Returns:
        Approver: A function that approves or rejects bash commands based on the allowed list and rules.
    """
    allowed_commands_set = set(allowed_commands)
    command_specific_rules = command_specific_rules or {}
    dangerous_chars = ["&", "|", ";", ">", "<", "`", "$", "(", ")"]

    def approve(tool_call: ToolCall, state: Optional[TaskState] = None) -> Approval:
        if tool_call.function != "bash":
            return Approval(
                decision="escalate",
                explanation=f"AllowListApprover only handles bash commands, got {tool_call.function}",
            )

        command = tool_call.arguments.get("cmd", "").strip()
        if not command:
            return Approval(decision="reject", explanation="Empty command")

        try:
            tokens = shlex.split(command)
        except ValueError as e:
            return Approval(
                decision="reject", explanation=f"Invalid command syntax: {str(e)}"
            )

        if any(char in command for char in dangerous_chars):
            return Approval(
                decision="reject",
                explanation=f"Command contains potentially dangerous characters: {', '.join(char for char in dangerous_chars if char in command)}",
            )

        base_command = tokens[0]

        # Handle sudo
        if base_command == "sudo":
            if not allow_sudo:
                return Approval(decision="reject", explanation="sudo is not allowed")
            if len(tokens) < 2:
                return Approval(decision="reject", explanation="Invalid sudo command")
            base_command = tokens[1]
            tokens = tokens[1:]

        if base_command not in allowed_commands_set:
            return Approval(
                decision="escalate",
                explanation=f"Command '{base_command}' is not in the allowed list. Allowed commands: {', '.join(allowed_commands_set)}",
            )

        # Check command-specific rules
        if base_command in command_specific_rules:
            allowed_subcommands = command_specific_rules[base_command]
            if len(tokens) > 1 and tokens[1] not in allowed_subcommands:
                return Approval(
                    decision="escalate",
                    explanation=f"{base_command} subcommand '{tokens[1]}' is not allowed. Allowed subcommands: {', '.join(allowed_subcommands)}",
                )

        return Approval(
            decision="approve", explanation=f"Command '{command}' is approved."
        )

    return approve

def tool_jsonable(tool_call: ToolCall | None = None) -> dict[str, Any]:
    def as_jsonable(value: Any) -> Any:
        return to_jsonable_python(value, exclude_none=True, fallback=lambda _x: None)

    tool_data = dict(
        id=tool_call.id,
        function=tool_call.function,
        arguments=tool_call.arguments,
        type=tool_call.type,
    )
    jsonable = as_jsonable(tool_data)
    return cast(dict[str, Any], deepcopy(jsonable))

def human_approver(agent_id: str) -> Approver:
    """
    Create an approver that submits the tool call for human approval via API.

    Args:
        agent_id (str): The unique identifier for the agent.

    Returns:
        Approver: A function that submits the tool call for human approval via API.
    """

    def approve(tool_call: ToolCall, state: Optional[TaskState] = None) -> Approval:
        print(
            f"Sending tool call request to user: {tool_call.function} {tool_call.arguments}"
        )

        # Serialize the state to JSON-compatible format
        state_json = state_jsonable(state)
        tool_json = tool_jsonable(tool_call)
        print(f"Tool JSON: {tool_json}")

        request = {
            "agent_id": agent_id,
            "task_state": state_json,
            "tool_choice": tool_json,
        }

        # Submit the approval request
        response = requests.post("http://localhost:8080/api/review", json=request)

        print(request)

        if response.status_code != 200:
            return Approval(
                decision="escalate",
                explanation=f"Failed to submit approval request: {response.text}",
            )

        review_id = response.json().get("id")
        if not review_id:
            return Approval(
                decision="escalate",
                explanation="Failed to get review ID from initial response",
            )

        # Poll the status endpoint until we get a response
        max_attempts = 120  # 5 minutes with 2-second intervals
        print(f"Waiting for human approval for review {review_id}")
        for _ in range(max_attempts):
            status_response = requests.get(
                f"http://localhost:8080/api/review/status?id={review_id}"
            )

            if status_response.status_code != 200:
                return Approval(
                    decision="escalate",
                    explanation=f"Failed to get approval status: {status_response.text}",
                )

            status_data = status_response.json()
            print(f"Status data: {status_data}")
            if status_data.get("status") == "pending":
                time.sleep(2)  # Wait 2 seconds before polling again
                continue

            if "decision" in status_data:
                decision = status_data["decision"]
                explanation = status_data.get(
                    "explanation", "Human provided no explanation."
                )

                # Check if there's a modified tool call
                if "modified_tool_call" in status_data:
                    modified_tool_call_data = status_data["modified_tool_call"]
                    try:
                        modified_tool_call = ToolCall.parse_obj(modified_tool_call_data)
                    except Exception as e:
                        return Approval(
                            decision="escalate",
                            explanation=f"Failed to parse modified tool call: {e}",
                        )
                    return Approval(
                        decision=decision,
                        explanation=explanation,
                        modified_tool_call=modified_tool_call,
                    )
                else:
                    return Approval(decision=decision, explanation=explanation)

            return Approval(
                decision="escalate",
                explanation=f"Unexpected response from status endpoint: {status_data}",
            )

        return Approval(
            decision="escalate",
            explanation="Timed out waiting for human approval",
        )

    return approve

def get_approval(
    approvers: list[Approver], tool_call: ToolCall, state: Optional[TaskState] = None
) -> tuple[bool, str, ToolCall]:
    """
    Get approval for a tool call using the list of approvers.

    Args:
        approvers (List[Approver]): A list of approvers to use in the approval process.
        tool_call (ToolCall): The tool call to be approved.
        state (Optional[TaskState]): The current task state, if available.

    Returns:
        Tuple[bool, str, ToolCall]: A tuple containing a boolean indicating approval status,
                                     a message explaining the decision, and the (possibly modified) ToolCall.
    """
    state = state or sample_state()
    for approver in approvers:
        approval = approver(tool_call, state)
        if approval.decision == "approve":
            print_approval_message(tool_call, approval.explanation)

            # Use the modified tool call if provided
            if approval.modified_tool_call:
                tool_call = approval.modified_tool_call
                print("[bold green]Using modified ToolCall provided by the approver.[/bold green]")

            return True, approval.explanation, tool_call
        elif approval.decision == "reject":
            print_rejection_message(tool_call, approval.explanation)
        elif approval.decision == "terminate":
            print_termination_message(approval.explanation)
            sys.exit(1)
        elif approval.decision == "escalate":
            print_escalation_message(tool_call, approval.explanation)

    final_message = "Rejected: No approver approved the tool call"
    print_rejection_message(tool_call, final_message)
    return False, final_message, tool_call

def print_approval_message(tool_call: ToolCall, reason: str):
    """
    Print an approval message for a tool call.

    Args:
        tool_call (ToolCall): The approved tool call.
        reason (str): The reason for approval.
    """
    with input_screen() as console:
        console.print(
            Panel.fit(
                f"Tool call approved:\nFunction: {tool_call.function}\nArguments: {tool_call.arguments}\nReason: {reason}",
                title="Tool Execution",
                subtitle="Approved",
            )
        )


def print_rejection_message(tool_call: ToolCall, reason: str):
    """
    Print a rejection message for a tool call.

    Args:
        tool_call (ToolCall): The rejected tool call.
        reason (str): The reason for rejection.
    """
    with input_screen() as console:
        console.print(
            Panel.fit(
                f"Tool call rejected:\nFunction: {tool_call.function}\nArguments: {tool_call.arguments}\nReason: {reason}",
                title="Tool Execution",
                subtitle="Rejected",
            )
        )


def print_escalation_message(tool_call: ToolCall, reason: str):
    """
    Print an escalation message for a tool call.

    Args:
        tool_call (ToolCall): The escalated tool call.
        reason (str): The reason for escalation.
    """
    with input_screen() as console:
        console.print(
            Panel.fit(
                f"Tool call escalated:\nFunction: {tool_call.function}\nArguments: {tool_call.arguments}\nReason: {reason}",
                title="Tool Execution",
                subtitle="Escalated",
            )
        )


def print_termination_message(reason: str):
    """
    Print a termination message.

    Args:
        reason (str): The reason for termination.
    """
    with input_screen() as console:
        console.print(
            Panel.fit(
                f"Execution terminated.\nReason: {reason}",
                title="Execution Terminated",
                subtitle="System Shutdown",
            )
        )


def print_tool_response_and_get_authorization(output: ModelOutput) -> bool:
    """
    Print the model's response and tool calls, and ask for user authorization.

    Args:
        output (ModelOutput): The model's output containing the response and tool calls.

    Returns:
        bool: True if the user authorizes the execution, False otherwise.
    """
    renderables: list[RenderableType] = []
    if output.message.content != "":
        renderables.append(
            Panel.fit(Markdown(str(output.message.content)), title="Textual Response")
        )

    renderables.append(
        Panel.fit(
            Group(
                *format_human_readable_tool_calls(output.message.tool_calls or []),
                fit=True,
            ),
            title="Tool Calls",
        )
    )
    with input_screen() as console:
        console.print(Panel.fit(Group(*renderables, fit=True), title="Model Response"))

        return Confirm.ask(
            "Do you FULLY understand these tool calls and approve their execution?"
        )


def format_human_readable_tool_calls(
    tool_calls: list[ToolCall],
) -> list[RenderableType]:
    """
    Format tool calls into human-readable renderable objects.

    Args:
        tool_calls (list[ToolCall]): List of tool calls to format.

    Returns:
        list[RenderableType]: A list of renderable objects representing the formatted tool calls.
    """
    output_renderables: list[RenderableType] = []
    for i, tool_call in enumerate(tool_calls):
        panel_contents = []
        for i, (argument, value) in enumerate(tool_call.arguments.items()):
            argument_contents = []
            match (tool_call.function, argument):
                case ("python", "code"):
                    argument_contents.append(
                        Syntax(
                            value,
                            "python",
                            theme="monokai",
                            line_numbers=True,
                        )
                    )
                case ("bash", "cmd"):
                    argument_contents.append(Syntax(value, "bash", theme="monokai"))
                case _:
                    argument_contents.append(value)
            panel_contents.append(
                Panel.fit(
                    Group(*argument_contents, fit=True),
                    title=f"Argument #{i}: [bold]{argument}[/bold]",
                )
            )
        if tool_call.parse_error is not None:
            output_renderables.append(f"Parse error: {tool_call.parse_error}")
        output_renderables.append(
            Panel.fit(
                Group(*panel_contents, fit=True),
                title=f"Tool Call #{i}: [bold]{tool_call.function}[/bold]",
            )
        )
    return output_renderables
