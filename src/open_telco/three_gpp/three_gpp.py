from textwrap import dedent

from dotenv import load_dotenv
from inspect_ai import task, Task
from inspect_ai.dataset import hf_dataset, Sample
from inspect_ai.scorer import accuracy, choice, stderr
from inspect_ai.solver import multiple_choice, system_message


load_dotenv()


SYSTEM_PROMPT = dedent("""
    You are a distinguished expert in telecommunication domain and you are skilled in understanding and classifying 3GPP technical documents. You will be classifying texts according to the working group they belong to.
    """)


def three_gpp_record_to_sample(record):
    return Sample(
        input=record["input"],
        choices=record["choices"],
        target=chr(65 + int(record["index"])),
    )


@task
def three_gpp() -> Task:
    dataset = hf_dataset(
        "eaguaida/three_gpp",
        sample_fields=three_gpp_record_to_sample,
        split="test",
    )
 
    solver = [system_message(SYSTEM_PROMPT), multiple_choice(cot=True)]
    
    return Task(
        dataset=dataset,
        solver=solver,
        scorer=choice(),
        metrics=[accuracy(),stderr()]
    )