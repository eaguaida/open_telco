from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice
from telecom_bench.scripts.utils import load_env, load_huggingface_dataset

# Load environment variables
load_env()


def teleqna_record_to_sample(record):
    """Convert TeleQnA record to Inspect Sample with choices and metadata."""
    return Sample(
        input=record["question"],
        choices=record["choices"],
        target=chr(65 + record["answer"]),
        metadata={"subject": record.get("subject")}
    )


@task
def teleqna(subject: str = "full") -> Task:
    """TeleQnA benchmark - Multiple choice telecommunications knowledge assessment.
    
    Args:
        subject: One of 'Research Overview', 'Standards Overview', 
                'Research Publications', 'Standards Specification', or 'full' (default)
    """
    dataset = load_huggingface_dataset(
        "netop/TeleQnA",
        sample_fields=teleqna_record_to_sample
    )
    
    if subject != "full":
        dataset = dataset.filter(
            lambda sample: sample.metadata.get("subject") == subject
        )
    
    return Task(
        dataset=dataset,
        solver=multiple_choice(cot=True),
        scorer=choice()
    )

