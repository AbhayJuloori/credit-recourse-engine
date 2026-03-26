"""
Push this project to HuggingFace Spaces as a Docker Space.

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login  (or set HF_TOKEN env var)

Run:
    python deploy_hf.py --space AbhayJuloori/credit-recourse-engine
"""

import argparse
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

def deploy(space_id: str):
    api = HfApi()
    
    # Verify authentication
    try:
        user = api.whoami()
        print(f"Logged in as: {user['name']}")
    except Exception as e:
        print(f"Not logged in: {e}")
        print("Run: python -c \"from huggingface_hub import login; login()\"")
        sys.exit(1)

    # Create Space if it doesn't exist
    print(f"\nCreating/updating Space: {space_id}")
    try:
        create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True,
            private=False,
        )
        print("Space created (or already exists).")
    except Exception as e:
        print(f"Space creation note: {e}")

    # Upload all project files
    root = Path(__file__).parent
    ignore_patterns = {
        ".venv", "venv", "__pycache__", ".git", ".pytest_cache",
        "*.pyc", "*.log", "*.csv", ".DS_Store",
    }

    files_to_upload = []
    for path in root.rglob("*"):
        if path.is_file():
            rel = path.relative_to(root)
            parts = rel.parts
            skip = False
            for part in parts:
                if any(
                    part == ign or (ign.startswith("*") and part.endswith(ign[1:]))
                    for ign in ignore_patterns
                ):
                    skip = True
                    break
            if not skip:
                files_to_upload.append((path, str(rel)))

    print(f"\nUploading {len(files_to_upload)} files…")
    for local_path, repo_path in files_to_upload:
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=repo_path,
                repo_id=space_id,
                repo_type="space",
            )
            print(f"  ✓ {repo_path}")
        except Exception as e:
            print(f"  ✗ {repo_path}: {e}")

    print(f"\nDone! Space: https://huggingface.co/spaces/{space_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", default="AbhayJuloori/credit-recourse-engine",
                        help="HuggingFace space ID (username/repo-name)")
    args = parser.parse_args()
    deploy(args.space)
