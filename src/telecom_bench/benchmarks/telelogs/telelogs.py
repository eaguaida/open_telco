from textwrap import dedent

from inspect_ai import Task, task  # noqa: E402
from inspect_ai.agent import Agent, agent, react, AgentPrompt  # noqa: E402
from inspect_ai.dataset import FieldSpec  # noqa: E402
from inspect_ai.scorer import pattern, accuracy, stderr  # noqa: E402
from telecom_bench.scripts.utils import load_env, load_huggingface_dataset  # type: ignore # noqa: E402

from telecom_bench.benchmarks.telelogs.utils import maj_at_k

# Load environment variables
load_env()

INSTRUCTIONS = dedent(
    """
    You are a telecommunications operator analyzing system logs to identify root causes of issues.
    You will receive log entries and need to determine which cause identifier best explains the problem.
    Think step by step through the log patterns, error codes, and sequence of events to reach your conclusion.
    
    CRITICAL: When submitting your answer with {submit}(), you MUST format it as \\boxed{C<value>} where <value> is the numeric identifier.
    - Use the format \\boxed{C<number>} - do NOT include spaces or additional text
    - Examples of correct format: "\\boxed{C5}", "\\boxed{C7}", "\\boxed{C2}", "\\boxed{C10}"
    - Examples of incorrect format: "\\boxed{5}", "\\boxed{cause 5}", "C5", "\\boxed{C 5}"
    """
)
@agent
def telelogs_agent(attempts: int = 1) -> Agent:
    return react(
        description="Telecommunications log analyst.",
        prompt=AgentPrompt(
            instructions=INSTRUCTIONS,
        ),
        tools=[],
        attempts=attempts,
    )


@task
def telelogs(epochs_count: int | None = None) -> Task:
    """TeleLogs benchmark with optional multi-epoch metrics.
    
    Args:
        epochs_count: Number of epochs to run for computing pass@1 and maj@N metrics
    """
    dataset = load_huggingface_dataset(
        "netop/TeleLogs",
        sample_fields=FieldSpec(
            input="question",
            target="answer",
        )
    )

    epochs = None
    task_metrics = None
    
    if epochs_count:
        epochs = epochs_count
        task_metrics = [accuracy(), stderr(), maj_at_k()]

    return Task(
        dataset=dataset,
        solver=telelogs_agent(),
        scorer=pattern(r"\\boxed\{\{?(.+?)\}?\}"),
        epochs=epochs,
        metrics=task_metrics,
    )


