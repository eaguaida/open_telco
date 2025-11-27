"""
Script to extract and transform the TeleLogs dataset from HuggingFace
"""
import os
import re
import sys
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import login
import json
import pandas as pd
import argparse

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from telecom_bench.prompt_transformer import transform_prompt

# Get HuggingFace token from environment variable
# Only check when running as main script, not when importing functions
HF_TOKEN = os.environ.get("HF_TOKEN")

def ensure_hf_login():
    """Ensure HuggingFace authentication is set up."""
    if not HF_TOKEN:
        raise ValueError(
            "HF_TOKEN environment variable not set. "
            "Please set it with your HuggingFace token:\n"
            "export HF_TOKEN=your_token_here"
        )

    # Login to HuggingFace
    try:
        login(token=HF_TOKEN, add_to_git_credential=False)
        print("Successfully logged in to HuggingFace")
    except Exception as e:
        print(f"Warning: Could not login to HuggingFace: {e}")

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
    # Try multiple patterns to find where the actual question starts

    # Pattern 1: C8 followed by \n\n (double newline)
    match = re.search(r'C8:.*?\n\n(.+)', question_text, re.DOTALL)
    if match:
        actual_question = match.group(1).strip()
        if len(actual_question) > 100:  # Sanity check: question should be substantial
            return actual_question

    # Pattern 2: Look for "Given:" which typically starts the actual question
    match = re.search(r'(Given:.+)', question_text, re.DOTALL | re.IGNORECASE)
    if match:
        actual_question = match.group(1).strip()
        if len(actual_question) > 100:
            return actual_question

    # Pattern 3: Look for "User plane" which starts the drive test data
    match = re.search(r'((?:User plane|drive test).+)', question_text, re.DOTALL | re.IGNORECASE)
    if match:
        actual_question = match.group(1).strip()
        if len(actual_question) > 100:
            return actual_question

    # If no pattern matches, log a detailed warning
    print(f"Warning: Could not extract question data. Text length: {len(question_text)}")
    print(f"First 200 chars: {question_text[:200]}")
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

def process_dataset(apply_markdown_kv: bool = False):
    """
    Load and process the TeleLogs dataset

    Args:
        apply_markdown_kv: If True, transform questions to Markdown-KV format
    """
    # Ensure authentication is set up
    ensure_hf_login()

    print("Loading dataset from HuggingFace...")
    try:
        # Load the dataset with authentication
        ds = load_dataset("eaguaida/telelogs", token=HF_TOKEN)
        print(f"Dataset loaded successfully!")
        print(f"Available splits: {list(ds.keys())}")

        # Get test split
        test_data = ds['test']
        print(f"\nTest split size: {len(test_data)}")

        # Convert to pandas for easier manipulation
        df = test_data.to_pandas()
        print(f"\nOriginal columns: {df.columns.tolist()}")

        # Show a sample before transformation
        print("\n=== Sample before transformation ===")
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

        # Verify extraction quality
        print("   Verifying extraction quality...")
        sample_question = df['question'].iloc[0]
        has_given = 'Given:' in sample_question or 'given:' in sample_question.lower()
        has_drive_test = 'drive test' in sample_question.lower() or 'user plane' in sample_question.lower()
        has_eng_params = 'engineering' in sample_question.lower() or 'eng parameters' in sample_question.lower()

        print(f"   - Contains 'Given:': {has_given}")
        print(f"   - Contains drive test data: {has_drive_test}")
        print(f"   - Contains engineering params: {has_eng_params}")

        if not (has_given and has_drive_test and has_eng_params):
            print("   ⚠ WARNING: Extraction may be incomplete!")
            print(f"   Sample length: {len(sample_question)} chars")
            print(f"   First 300 chars:\n{sample_question[:300]}")

        # 4. Optionally transform to Markdown-KV format
        if apply_markdown_kv:
            print("4. Transforming questions to Markdown-KV format...")
            df['question'] = df['question'].apply(transform_prompt)
            print("   Markdown-KV transformation applied!")

            # Verify transformation quality
            print("   Verifying transformation...")
            sample_transformed = df['question'].iloc[0]
            checks = {
                'has_domain_rules': '# Domain Rules' in sample_transformed,
                'has_drive_test': '# Drive Test Data' in sample_transformed,
                'has_engineering': '# Engineering Parameters' in sample_transformed,
                'has_relationships': '# Data Relationships' in sample_transformed,
                'has_code_blocks': '```' in sample_transformed
            }

            for check_name, passed in checks.items():
                status = "✓" if passed else "✗"
                print(f"   {status} {check_name}")

            if not all(checks.values()):
                print("   ⚠ WARNING: Transformation may have failed!")
                print(f"   Sample length: {len(sample_transformed)} chars")
                print(f"   First 500 chars:\n{sample_transformed[:500]}")

        # Show a sample after transformation
        print("\n=== Sample after transformation ===")
        print(f"Question: {df['question'].iloc[0]}")
        print(f"Answer: {df['answer'].iloc[0]}")
        print(f"Choices (first 2): {df['choices'].iloc[0][:2]}")
        print(f"Number of choices: {len(df['choices'].iloc[0])}")

        # Save to different formats
        print("\n=== Saving dataset ===")

        # Add suffix for Markdown-KV format
        suffix = "_mkv" if apply_markdown_kv else ""

        # Save as CSV
        csv_path = f"telelogs_markdown/telelogs_test{suffix}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved to {csv_path}")

        # Save as JSON
        json_path = f"telelogs_markdown/telelogs_test{suffix}.json"
        df.to_json(json_path, orient='records', indent=2)
        print(f"Saved to {json_path}")

        # Save as parquet (more efficient for large datasets)
        parquet_path = f"telelogs_markdown/telelogs_test{suffix}.parquet"
        df.to_parquet(parquet_path, index=False)
        print(f"Saved to {parquet_path}")

        print("\n=== Processing complete! ===")
        print(f"Total samples processed: {len(df)}")
        if apply_markdown_kv:
            print(f"Format: Markdown-KV (improved LLM comprehension)")
        print("\nNote: README.md with dataset card already exists in telelogs_markdown/")
        print("Upload to HuggingFace with: python upload_telelogs.py")

        return df

    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and process TeleLogs dataset from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract in standard format
  python extract_telelogs.py

  # Extract with Markdown-KV transformation (improved LLM comprehension)
  python extract_telelogs.py --markdown-kv

The Markdown-KV format achieves ~60% accuracy vs ~41-44% for pipe-delimited
format in LLM table understanding benchmarks.
        """
    )

    parser.add_argument(
        '--markdown-kv',
        action='store_true',
        help='Transform questions to Markdown-KV format for improved LLM comprehension'
    )

    args = parser.parse_args()
    df = process_dataset(apply_markdown_kv=args.markdown_kv)
