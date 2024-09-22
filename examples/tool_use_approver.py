from typing import Any, List
from inspect_ai import Task, task, eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import ToolCall, Tool, tool
from inspect_ai.util._approver import ApprovalResult
from inspect_ai.util import sandbox
from inspect_ai import eval


def create_bash_approval_function(whitelist_commands: List[str], on_failure: ApprovalResult = ApprovalResult.ESCALATE):
    """Create an approval function for bash commands.

    This function creates an approval function that only allows commands
    in the provided whitelist.

    Args:
        whitelist_commands (List[str]): List of allowed command prefixes.
        on_failure (ApprovalResult): The result to return if the command
            is not in the whitelist. Defaults to ApprovalResult.ESCALATE.

    Returns:
        A function that takes a ToolCall and a tool, and returns an ApprovalResult.
    """
    def bash_approval_function(call: ToolCall, tool: Any) -> ApprovalResult:
        """
        Only allow commands in the whitelist.
        """
        command = call.arguments['cmd']
        if not any(command.startswith(whitelist_command) for whitelist_command in whitelist_commands):
            return on_failure
        return ApprovalResult.APPROVE
    return bash_approval_function


def bash(timeout: int | None = None, user: str | None = None) -> Tool:
    """Bash shell command execution tool.

    Execute bash shell commands using a sandbox environment (e.g. "docker").

    Args:
      timeout (int | None): Timeout (in seconds) for command.
      user (str | None): User to execute commands as.

    Returns:
      String with command output (stdout) or command error (stderr).
    """

    async def execute(cmd: str) -> str:
        """
        Use this function to execute bash commands.

        Args:
          cmd (str): The bash command to execute.

        Returns:
          The output of the command.
        """
        # execute the command
        result = await sandbox().exec(
            cmd=["bash", "-c", cmd], timeout=timeout, user=user
        )
        # return output (including stderr if any)
        output = ""
        if result.stderr:
            output = f"{result.stderr}\n"
        return f"{output}{result.stdout}"

    return execute


@task
def bash_task():
    dataset = [
        Sample(
            input=(
                "Please list the files in the /usr/bin directory. "
                + "Is there a file named 'python3' in the directory?"
            ),
            target=["Yes"],
        ),
        Sample(input="Please write 'bar' to a file named 'foo.txt'.")
    ]

    # Create the approval function with the desired whitelist commands
    approval_function = create_bash_approval_function(["ls", "cat", "apt-get install", "pip install"])

    # Use the tool decorator with the approval function
    bash_tool = tool(bash, approval_function=approval_function)

    return Task(
        dataset=dataset,
        plan=[
            use_tools(bash_tool()),
            generate(),
        ],
        sandbox="local",
        scorer=includes(),
    )


if __name__ == "__main__":
    from inspect_ai import eval
    tasks = [bash_task()]
    eval(tasks=tasks, model='openai/gpt-4o-mini', limit=2, log_buffer=1)
