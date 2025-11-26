---
language:
- en
license: apache-2.0
task_categories:
- multiple-choice
- question-answering
tags:
- telecommunications
- 5G
- network-analysis
- root-cause-analysis
pretty_name: TeleLogs (Processed MCQ Format)
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
  - split: test
    path: telelogs_test.parquet
---

# TeleLogs Dataset (Processed MCQ Format)

This dataset has been extracted from the original [netop/TeleLogs](https://huggingface.co/datasets/netop/TeleLogs) dataset and processed into multiple-choice question (MCQ) format for easier evaluation.

## Dataset Description

TeleLogs is a telecommunications log analysis benchmark where models must identify the root cause of network issues from 5G wireless network drive-test data and engineering parameters.

### Processed Format

This version has been restructured for MCQ evaluation with the following improvements:

- **Clean separation** of question data, choices, and instructions
- **0-based indexing** for answers (0-7 instead of 1-8)
- **Extracted choices** as a proper array field
- **Removed template text** from questions

## Dataset Structure

### Data Instances

Each instance contains:
- `question`: The actual network data (drive-test logs, engineering parameters, tables)
- `choices`: Array of 8 possible root causes
- `answer`: Integer index (0-7) indicating the correct choice

Example:
```json
{
  "question": "Given:\n- The default electronic downtilt value is 255...\n\nUser plane drive test data as follows：\n\n<tables>...",
  "choices": [
    "The serving cell's downtilt angle is too large, causing weak coverage at the far end.",
    "The serving cell's coverage distance exceeds 1km, resulting in over-shooting.",
    ...
  ],
  "answer": 3
}
```

### Data Fields

- `question` (string): The network data and parameters to analyze. Typically starts with "Given:" and includes:
  - Configuration parameters
  - Drive test data tables
  - Engineering parameters tables

- `choices` (list of strings): Array of exactly 8 possible root causes

- `answer` (integer): The correct answer index (0-7), where:
  - 0 = First choice (originally C1)
  - 1 = Second choice (originally C2)
  - ...
  - 7 = Eighth choice (originally C8)

### Data Splits

|     | test |
|-----|------|
| TeleLogs | 864 |

## Transformations Applied

This processed version applies three key transformations:

### 1. Answer Column
- **Original**: `C1`, `C2`, ..., `C8`
- **Processed**: `0`, `1`, ..., `7`
- Converted to 0-based indexing to match array positions

### 2. Choices Column (New)
- Extracted 8 choices cleanly from the original question text
- Each choice stored as array element
- For C8, stops at `\n\n` separator before actual question data

### 3. Question Column
- **Removed**: Template instructions (e.g., "Analyze the 5G wireless network...")
- **Removed**: Choice listings (C1-C8 with their text)
- **Kept**: Only the actual question data after `\n\n`
- Typically starts with "Given:" and includes all data tables

## Dataset Creation

### Source Data

Original dataset: [netop/TeleLogs](https://huggingface.co/datasets/netop/TeleLogs)

### Processing Pipeline

1. Load original TeleLogs dataset from HuggingFace
2. Extract test split
3. Parse and separate choices from question text using regex
4. Convert answer format from C1-C8 to 0-7
5. Remove instruction template and choice listings
6. Export to multiple formats (CSV, JSON, Parquet)

### Processing Scripts

The processing scripts are available at: [Telecom-Bench](https://github.com/eaguaida/Telecom-Bench)

## Usage

### Loading with Hugging Face Datasets

```python
from datasets import load_dataset

dataset = load_dataset("eaguaida/telelogs")
```

### Using with Inspect AI

```python
from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

def telelogs_record_to_sample(record):
    return Sample(
        input=record["question"],
        choices=record["choices"],
        target=chr(65 + record["answer"]),  # Convert 0->A, 1->B, etc.
    )

# Load and evaluate
dataset = load_dataset("eaguaida/telelogs", sample_fields=telelogs_record_to_sample)
task = Task(dataset=dataset, solver=multiple_choice(), scorer=choice())
```

## File Formats

The dataset is available in multiple formats:

- **`telelogs_test.parquet`**: Parquet format (recommended, most efficient)
- **`telelogs_test.json`**: JSON format (human-readable)
- **`telelogs_test.csv`**: CSV format (note: choices stored as JSON string)

## Licensing

This dataset maintains the same license as the original TeleLogs dataset.

## Citation

If you use this dataset, please cite the original TeleLogs dataset:

```bibtex
@dataset{telelogs2024,
  title={TeleLogs: Telecommunications Log Analysis Dataset},
  author={Original Authors},
  year={2024},
  publisher={HuggingFace},
  url={https://huggingface.co/datasets/netop/TeleLogs}
}
```

## Contact

For questions or issues with this processed version, please open an issue at [Telecom-Bench](https://github.com/eaguaida/Telecom-Bench).
