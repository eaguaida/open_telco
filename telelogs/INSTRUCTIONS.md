# TeleLogs Dataset Extraction Instructions

## Problem
The current environment has network restrictions that block access to HuggingFace. The dataset cannot be downloaded automatically from `netop/TeleLogs`.

## Solutions

### Option 1: Run Locally (Recommended)

Run the extraction script on your local machine where you have internet access to HuggingFace:

```bash
# Make sure you're in the project root
cd /path/to/Telecom-Bench

# Install dependencies
pip install -e .

# Set your HuggingFace token as an environment variable
export HF_TOKEN=your_huggingface_token_here

# Run the extraction script
python extract_telelogs.py
```

The script will:
1. Download the TeleLogs dataset from HuggingFace (using the provided token)
2. Process the test split
3. Transform the data:
   - Remove 'C' prefix from answers (C4 → 4)
   - Extract 8 choices into an array
   - Clean the questions by removing template text
4. Save to `telelogs/` folder in multiple formats (CSV, JSON, Parquet)

### Option 2: Manual Download and Transform

1. Download the dataset manually from https://huggingface.co/datasets/netop/TeleLogs
2. Place the files in this `telelogs` folder
3. Run a simplified transformation script (to be created if needed)

### Option 3: Use Alternative Environment

Run this in an environment without proxy restrictions:
- Google Colab
- Local Jupyter notebook
- Any machine with direct internet access

## Dataset Transformation Details

The script performs these transformations on the test split:

1. **Answer column**: Strips 'C' prefix and converts to 0-based index
   - Before: `C1` -> After: `0`
   - Before: `C4` -> After: `3`
   - Before: `C8` -> After: `7`
   - This matches the array indexing of the choices

2. **Choices column** (new): Extracts 8 choices cleanly from question text
   - Parses choices formatted as `C1: content`, `C2: content`, ..., `C8: content`
   - For C8, stops at `\n\n` which separates the choice from actual question data
   - Creates array: `["choice1", "choice2", ..., "choice8"]`
   - Choices are indexed 0-7 in the array

3. **Question column**: Extracts actual question data only
   - **Removes** the instruction template completely (e.g., "Analyze the 5G wireless network...")
   - **Removes** the choice listing (C1-C8)
   - **Keeps only** the actual question data after `\n\n` following C8
   - Typically starts with "Given:" and includes all data tables

## Output Files

After successful extraction, you'll have:
- `telelogs_test.csv` - CSV format
- `telelogs_test.json` - JSON format
- `telelogs_test.parquet` - Parquet format (most efficient)
- `README.md` - Dataset documentation

## HuggingFace Authentication

You'll need a HuggingFace token with access to the `netop/TeleLogs` dataset. Set it as an environment variable:

```bash
export HF_TOKEN=your_huggingface_token_here
```

Alternatively, you can use `huggingface-cli login` if you have the HuggingFace CLI installed.

## Next Steps

Once you've extracted the dataset:
1. Commit the processed files to the repository
2. The data will be available for the LLM evaluation pipeline
3. You can work with the dataset in MCQ format
