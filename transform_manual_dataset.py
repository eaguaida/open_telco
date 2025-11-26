"""
Alternative script to transform manually downloaded TeleLogs dataset
Use this if you have the dataset files locally but can't use load_dataset()
"""
import re
import json
import pandas as pd
from pathlib import Path

def extract_choices_from_question(question_text):
    """
    Extract the 8 choices (C1-C8) from the question text.
    For C8, stops at \n\n which separates the choice from the actual question data.
    Returns a list of 8 choice contents.
    """
    choices = []

    # Pattern to match C1: through C8:
    # For C1-C7: content until next C digit or C8
    # For C8: content until \n\n (which marks start of actual question)
    pattern = r'C(\d):\s*(.*?)(?=\nC\d:|\n\n|$)'

    matches = re.findall(pattern, question_text, re.DOTALL)

    # Sort by choice number and extract just the content
    sorted_matches = sorted(matches, key=lambda x: int(x[0]))
    choices = [match[1].strip() for match in sorted_matches]

    # Ensure we have exactly 8 choices
    if len(choices) != 8:
        print(f"Warning: Found {len(choices)} choices instead of 8")

    return choices

def extract_actual_question(question_text):
    """
    Extract the actual question data (everything after \n\n following C8).
    This typically starts with "Given:" and contains the data tables.
    Removes the template instruction and choices completely.
    """
    # Find where C8 ends and the actual question begins (marked by \n\n)
    # Look for C8: ... \n\n<actual question>
    match = re.search(r'C8:.*?\n\n(.+)', question_text, re.DOTALL)

    if match:
        # Extract everything after the \n\n
        actual_question = match.group(1).strip()
        return actual_question
    else:
        # Fallback: if pattern not found, return empty or original
        print("Warning: Could not find actual question data after C8")
        return question_text.strip()

def transform_answer(answer):
    """
    Remove 'C' prefix from answer and convert to 0-based index.
    Example: C1 -> 0, C4 -> 3, C8 -> 7
    Since choices array is 0-indexed, we subtract 1.
    """
    if isinstance(answer, str) and answer.startswith('C'):
        return int(answer[1:]) - 1
    return answer

def process_manual_dataset(input_path, output_dir="telelogs"):
    """
    Process a manually downloaded dataset file

    Args:
        input_path: Path to the dataset file (JSON, JSONL, or CSV)
        output_dir: Directory to save processed files
    """
    print(f"Loading dataset from {input_path}...")

    # Load the data based on file type
    input_path = Path(input_path)
    if input_path.suffix == '.jsonl':
        data = []
        with open(input_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        df = pd.DataFrame(data)
    elif input_path.suffix == '.json':
        df = pd.read_json(input_path)
    elif input_path.suffix == '.csv':
        df = pd.read_csv(input_path)
    elif input_path.suffix == '.parquet':
        df = pd.read_parquet(input_path)
    else:
        raise ValueError(f"Unsupported file type: {input_path.suffix}")

    print(f"Loaded {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")

    # Check if we need to filter for test split
    if 'split' in df.columns:
        print(f"Filtering for test split...")
        df = df[df['split'] == 'test'].copy()
        print(f"Test split size: {len(df)}")

    # Show a sample before transformation
    print("\n=== Sample before transformation ===")
    if len(df) > 0:
        print(f"Question (first 500 chars): {df['question'].iloc[0][:500]}...")
        print(f"Answer: {df['answer'].iloc[0]}")

    # Apply transformations
    print("\n=== Applying transformations ===")

    # 1. Transform answer column
    print("1. Transforming answer column...")
    df['answer'] = df['answer'].apply(transform_answer)

    # 2. Extract choices into new column
    print("2. Extracting choices...")
    df['choices'] = df['question'].apply(extract_choices_from_question)

    # 3. Extract actual question data (after C8 and \n\n)
    print("3. Extracting actual question data...")
    df['question'] = df['question'].apply(extract_actual_question)

    # Show a sample after transformation
    print("\n=== Sample after transformation ===")
    if len(df) > 0:
        print(f"Question: {df['question'].iloc[0]}")
        print(f"Answer: {df['answer'].iloc[0]}")
        print(f"Choices (first 2): {df['choices'].iloc[0][:2]}")
        print(f"Number of choices: {len(df['choices'].iloc[0])}")

    # Save to different formats
    print("\n=== Saving dataset ===")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Save as CSV
    csv_path = output_dir / "telelogs_test.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}")

    # Save as JSON
    json_path = output_dir / "telelogs_test.json"
    df.to_json(json_path, orient='records', indent=2)
    print(f"Saved to {json_path}")

    # Save as parquet
    parquet_path = output_dir / "telelogs_test.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Saved to {parquet_path}")

    # Save a README
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write("# TeleLogs Dataset (Processed)\n\n")
        f.write("This dataset has been extracted from HuggingFace and processed into MCQ format.\n\n")
        f.write("## Transformations Applied\n\n")
        f.write("1. **Answer column**: Removed 'C' prefix, keeping only the number (C4 → 4)\n")
        f.write("2. **Choices column**: Extracted 8 choices from questions into an array\n")
        f.write("3. **Question column**: Removed template text and choice options\n\n")
        f.write(f"## Dataset Info\n\n")
        f.write(f"- Number of samples: {len(df)}\n")
        f.write(f"- Columns: {', '.join(df.columns.tolist())}\n\n")
        f.write(f"## Files\n\n")
        f.write(f"- `telelogs_test.csv`: CSV format\n")
        f.write(f"- `telelogs_test.json`: JSON format\n")
        f.write(f"- `telelogs_test.parquet`: Parquet format (recommended for large datasets)\n")
    print(f"Saved README to {readme_path}")

    print("\n=== Processing complete! ===")
    print(f"Total samples processed: {len(df)}")

    return df

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python transform_manual_dataset.py <input_file>")
        print("Example: python transform_manual_dataset.py test.json")
        print("\nSupported formats: JSON, JSONL, CSV, Parquet")
        sys.exit(1)

    input_file = sys.argv[1]
    df = process_manual_dataset(input_file)
