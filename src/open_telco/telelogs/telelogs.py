from dotenv import load_dotenv
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import choice, accuracy, stderr
from inspect_ai.solver import multiple_choice

from open_telco.telelogs.utils import maj_at_k


load_dotenv()


def telelogs_record_to_sample(record):
    return Sample(
        input=record["question"],
        choices=record["choices"],
        target=chr(65 + record["answer"]),  # Convert 0->A, 1->B, 2->C, etc.
        metadata={}
    )


@task
def telelogs() -> Task:
    dataset = hf_dataset(
        "eaguaida/telelogs",
        sample_fields=telelogs_record_to_sample,
        split="test",
    )

    return Task(
        dataset=dataset,
        solver=multiple_choice(cot=False),
        scorer=choice(),
        metrics=[accuracy(),stderr(), maj_at_k()]
    )