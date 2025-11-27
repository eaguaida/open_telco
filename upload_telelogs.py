"""
Script to upload the processed TeleLogs dataset to HuggingFace
"""
import os
import argparse
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

def upload_dataset(markdown_kv: bool = False, repo_id: str = "eaguaida/telelogs"):
    """Upload the processed TeleLogs dataset to HuggingFace (JSON + README only).

    Args:
        markdown_kv: If True, upload the Markdown-KV formatted version (_mkv files)
        repo_id: HuggingFace repository ID to upload to
    """

    # Initialize HuggingFace API
    api = HfApi(token=HF_TOKEN)

    # Path to the processed dataset folder
    dataset_folder = Path("telelogs")

    if not dataset_folder.exists():
        raise FileNotFoundError(
            f"Dataset folder '{dataset_folder}' not found. "
            "Please run extract_telelogs.py first to create the processed dataset."
        )

    # Determine which file to upload based on format
    suffix = "_mkv" if markdown_kv else ""
    json_file = dataset_folder / f"telelogs_test{suffix}.json"
    readme_file = dataset_folder / "README.md"

    if not json_file.exists():
        raise FileNotFoundError(
            f"JSON file not found: {json_file}\n"
            f"Please run: python extract_telelogs.py{' --markdown-kv' if markdown_kv else ''}"
        )

    if not readme_file.exists():
        print(f"Warning: README.md not found at {readme_file}")
        print("Dataset card will not be displayed properly on HuggingFace.")

    format_name = "Markdown-KV" if markdown_kv else "Standard"
    print("=" * 60)
    print(f"Uploading TeleLogs dataset to HuggingFace ({format_name} format)")
    print("=" * 60)
    print(f"Source folder: {dataset_folder.absolute()}")
    print(f"Target repository: {repo_id}")
    print(f"Repository type: dataset")
    print(f"\nFiles to upload:")
    print(f"  - {json_file.name} (data)")
    print(f"  - README.md (dataset card with YAML metadata)")
    print()

    try:
        # Upload only JSON and README
        files_to_upload = [
            (json_file.name, str(json_file)),
            ("README.md", str(readme_file))
        ]

        for path_in_repo, local_path in files_to_upload:
            if Path(local_path).exists():
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=path_in_repo,
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=f"Upload {path_in_repo} ({format_name} format)"
                )
                print(f"✅ Uploaded: {path_in_repo}")

        print()
        print("✅ Dataset uploaded successfully!")
        print()
        print(f"View your dataset at: https://huggingface.co/datasets/{repo_id}")
        print()
        print("The dataset card should now display with:")
        print("  - Proper YAML metadata")
        print("  - Dataset description and structure")
        print("  - Usage examples")
        if markdown_kv:
            print("  - Markdown-KV formatted questions for improved LLM comprehension")

    except Exception as e:
        print(f"❌ Error uploading dataset: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload TeleLogs dataset to HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload standard format to eaguaida/telelogs
  python upload_telelogs.py

  # Upload Markdown-KV format to eaguaida/telelogs
  python upload_telelogs.py --markdown-kv

  # Upload to custom repository
  python upload_telelogs.py --markdown-kv --repo-id myuser/my-telelogs
        """
    )

    parser.add_argument(
        '--markdown-kv',
        action='store_true',
        help='Upload the Markdown-KV formatted version (_mkv files)'
    )

    parser.add_argument(
        '--repo-id',
        default='eaguaida/telelogs',
        help='HuggingFace repository ID (default: eaguaida/telelogs)'
    )

    args = parser.parse_args()
    upload_dataset(markdown_kv=args.markdown_kv, repo_id=args.repo_id)
