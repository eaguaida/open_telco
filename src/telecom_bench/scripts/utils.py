import os
from pathlib import Path
from typing import Callable, Any, Union
from inspect_ai.dataset import FieldSpec, hf_dataset, Sample


def load_env():
    """Load environment variables from .env file."""
    candidates = (
        Path(__file__).resolve().parents[1] / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    )
    for env_path in candidates:
        if not env_path.exists():
            continue
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())
        return


def load_huggingface_dataset(
    dataset_name: str, 
    sample_fields: Union[FieldSpec, Callable[[Any], Sample]]
):
    """Load a HuggingFace dataset with authentication."""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN environment variable not set. "
            f"Please set it to access the gated {dataset_name} dataset. "
            "Get your token from: https://huggingface.co/settings/tokens"
        )
    
    return hf_dataset(
        dataset_name,
        split="test",
        token=hf_token,
        sample_fields=sample_fields,
    )

