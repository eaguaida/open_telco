"""
Script to upload the processed TeleLogs dataset to HuggingFace
"""
import os
from huggingface_hub import HfApi
from pathlib import Path

# Get HuggingFace token from environment variable
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN environment variable not set. "
        "Please set it with your HuggingFace token:\n"
        "export HF_TOKEN=your_token_here"
    )

def upload_dataset():
    """Upload the processed TeleLogs dataset to HuggingFace (JSON + README only)."""

    # Initialize HuggingFace API
    api = HfApi(token=HF_TOKEN)

    # Path to the processed dataset folder
    dataset_folder = Path("telelogs")

    if not dataset_folder.exists():
        raise FileNotFoundError(
            f"Dataset folder '{dataset_folder}' not found. "
            "Please run extract_telelogs.py first to create the processed dataset."
        )

    # Check if required files exist
    json_file = dataset_folder / "telelogs_test.json"
    readme_file = dataset_folder / "README.md"

    if not json_file.exists():
        raise FileNotFoundError(
            f"JSON file not found: {json_file}\n"
            "Please run extract_telelogs.py first."
        )

    if not readme_file.exists():
        print(f"Warning: README.md not found at {readme_file}")
        print("Dataset card will not be displayed properly on HuggingFace.")

    print("=" * 60)
    print("Uploading TeleLogs dataset to HuggingFace")
    print("=" * 60)
    print(f"Source folder: {dataset_folder.absolute()}")
    print(f"Target repository: eaguaida/telelogs")
    print(f"Repository type: dataset")
    print(f"\nFiles to upload:")
    print(f"  - telelogs_test.json (data)")
    print(f"  - README.md (dataset card with YAML metadata)")
    print()

    try:
        # Upload only JSON and README
        files_to_upload = [
            ("telelogs_test.json", str(json_file)),
            ("README.md", str(readme_file))
        ]

        for path_in_repo, local_path in files_to_upload:
            if Path(local_path).exists():
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=path_in_repo,
                    repo_id="eaguaida/telelogs",
                    repo_type="dataset",
                    commit_message=f"Upload {path_in_repo}"
                )
                print(f"✅ Uploaded: {path_in_repo}")

        print()
        print("✅ Dataset uploaded successfully!")
        print()
        print("View your dataset at: https://huggingface.co/datasets/eaguaida/telelogs")
        print()
        print("The dataset card should now display with:")
        print("  - Proper YAML metadata")
        print("  - Dataset description and structure")
        print("  - Usage examples")

    except Exception as e:
        print(f"❌ Error uploading dataset: {e}")
        raise

if __name__ == "__main__":
    upload_dataset()
