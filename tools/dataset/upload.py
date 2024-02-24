import argparse
import os
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--upload-path", type=str)

    return parser.parse_args()


args = arg_parse()

repo_name: str = args.upload_path
branch_name: str = "main"
try:
    create_repo(repo_name, repo_type="dataset", private=True)
except Exception as e:
    print(f"repo {repo_name} already exists! error: {e}")
    pass

files = os.listdir(args.input_dir)

api = HfApi()
if branch_name != "main":
    try:
        api.create_branch(
            repo_id=repo_name,
            repo_type="dataset",
            branch=branch_name,
        )
    except Exception:
        print(f"branch {branch_name} already exists, try again...")
print(f"to upload: {files}")

for file in files:
    print(f"Uploading {file} to branch {branch_name}...")
    api.upload_file(
        path_or_fileobj=os.path.join(args.input_dir, file),
        path_in_repo=file,
        repo_id=repo_name,
        repo_type="dataset",
        commit_message=f"Upload {file}",
        revision=branch_name,
    )
    print(f"Successfully uploaded {file} !")
