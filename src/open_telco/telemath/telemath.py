from dotenv import load_dotenv
from inspect_ai import Task, task
from inspect_ai.agent import Agent, AgentPrompt, agent, react
from inspect_ai.dataset import FieldSpec, hf_dataset
from inspect_ai.scorer import match
from inspect_ai.tool import bash, python    


load_dotenv()


@agent
def telecom_agent(attempts=1) -> Agent:
    PROMPT = dedent("""
    You are a telecommunications expert solving mathematical problems.
    You have several tools available to help with finding the answer. You will
    see the result of tool calls right after sending the message. If you need
    to perform multiple actions, you can always send more messages with additional
    tool calls. Do some reasoning before your actions, describing what tool calls
    you are going to use and how they fit into your plan.

    CRITICAL: When submitting your answer with {submit}(), you MUST format it as a decimal number with full precision.
    - Include ALL decimal places - do NOT round to 2 decimals unless that's the exact answer
    - Examples of correct format: "2.0", "111.23", "15.456789", "1000.0"
    """)

    return react(
        description="Telecommunication Operator.",
        prompt=AgentPrompt(
            instructions=PROMPT,
        ),
        tools=[bash(), python()],
        attempts=attempts
    )


@task
def telemath(difficulty: str = "full"):
    """TeleMath benchmark with configurable difficulty.
    
    Args:
        difficulty: One of 'basic', 'intermediate', 'advanced', or 'full' (default)
    """
    dataset = hf_dataset(
        "netop/TeleMath",
        sample_fields=FieldSpec(
            input="question",
            target="answer",
            metadata=["category", "tags", "difficulty"],
        ),
        split="test",
    )
    
    if difficulty != "full":
        dataset = dataset.filter(
            lambda sample: sample.metadata.get("difficulty") == difficulty
        )
    
    return Task(
        dataset=dataset,
        solver=telecom_agent(),
        scorer=match(),
        sandbox=("docker", "../../../data/sandboxes/coding_env/compose.yaml")
    )

