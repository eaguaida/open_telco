import re
from textwrap import dedent

from dotenv import load_dotenv
from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, hf_dataset
from inspect_ai.scorer import scorer, accuracy, stderr, Score, Target
from inspect_ai.solver import generate, system_message, TaskState
 

load_dotenv()

SYSTEM_PROMPT = dedent(r"""
    You are an expert problem solver. Your task is to solve numerical exercises by following these guidelines:
    1.  **Understand the Goal:** Clearly identify what the problem is asking you to find, paying close attention to the required units for the final answer.
    2.  **Reason Step-by-Step:** Provide a clear, sequential reasoning process. Explain the formulas, principles, or logic used in each step. Show intermediate calculations if they clarify your thought process. The detailed structure of your sub-steps is up to you, as long as the reasoning is sound and easy to follow.
    3.  **Unit Management:**
        *   Track units throughout your calculations.
        *   **Crucially, ensure your final numerical answer is converted to the specific units requested in the problem statement.** If intermediate calculations result in a different unit, perform a final conversion step.
        *   State the unit of the final answer clearly in your explanatory text *before* the boxed answer.
    4.  **Final Numerical Answer Format:**
        *   The final answer must be a single numerical value (integer or float).
        *   Present this numerical value exclusively within the `\$\boxed{{...}}\$` format.
        *   **CRITICAL:** The `\$\boxed{{...}}\$` block must contain *only* the number. No text, no units, no labels (e.g., NOT `\$\boxed{{Result: 50}}\$` or `\$\boxed{{50 \text{{ mA}}}}\$`, but `\$\boxed{{50}}\$`).
    """)


def parse_answer(response: str) -> str:
    matches = re.findall(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', response)
    if matches:
        pred = matches[-1].strip()
    else:
        return ""
    pred = re.sub(r"\n\s*", "", pred).lstrip(":").rstrip("./")
    return pred


@scorer(metrics=[accuracy(), stderr()])
def telemath_scorer():
    async def score(state: TaskState, target: Target):
        answer = state.output.completion
        parsed = parse_answer(answer)
        correct = parsed == target.text
        return Score(value=correct, answer=parsed)
    return score


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
    
    solver = [system_message(SYSTEM_PROMPT), generate()]
    
    return Task(
        dataset=dataset,
        solver=solver,
        scorer=telemath_scorer(),
    )
