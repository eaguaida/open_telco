from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice
from telecom_bench.scripts.utils import load_env, load_huggingface_dataset

# Load environment variables
load_env()


def telelogs_record_to_sample(record):
    """Convert TeleLogs record to Inspect Sample with choices and metadata.

    The record should have:
    - question: The actual question data (tables, parameters, etc.)
    - choices: Array of 8 choice strings
    - answer: 0-based index (0-7) indicating the correct choice
    """
    return Sample(
        input=record["question"],
        choices=record["choices"],
        target=chr(65 + record["answer"]),  # Convert 0->A, 1->B, 2->C, etc.
        metadata={}
    )


@task
def telelogsmcq() -> Task:
    """TeleLogs benchmark - Multiple choice telecommunications log analysis.

    This version uses the MCQ format with extracted choices, where:
    - The question contains only the data (tables and parameters)
    - 8 choices are provided as options A-H
    - The model selects the most likely root cause
    """
    dataset = load_huggingface_dataset(
        "netop/TeleLogs",
        sample_fields=telelogs_record_to_sample
    )

    return Task(
        dataset=dataset,
        solver=multiple_choice(cot=False),
        scorer=choice()
    )
