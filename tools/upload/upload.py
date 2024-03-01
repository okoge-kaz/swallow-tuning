import os
import argparse

from huggingface_hub import HfApi, create_repo


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt-path", type=str)
parser.add_argument("--repo-name", type=str)
parser.add_argument("--branch-name", type=str, default="main")
args = parser.parse_args()

converted_ckpt: str = args.ckpt_path
repo_name: str = args.repo_name
branch_name: str = args.branch_name
try:
    create_repo(repo_name, repo_type="model", private=True)
except Exception as e:
    print(f"repo {repo_name} already exists! error: {e}")
    pass

files = os.listdir(converted_ckpt)
# filter
upload_files = []
for file_name in files:
    if file_name.endswith(".safetensors"):
        upload_files.append(file_name)
    elif file_name.endswith("config.json"):
        # config.json, generation_config.json
        upload_files.append(file_name)
    elif file_name.startswith("tokenizer"):
        # tokenizer_config.json, tokenizer.json, tokenizer.model
        upload_files.append(file_name)
    elif file_name == "model.safetensors.index.json":
        upload_files.append(file_name)
    elif file_name == "special_tokens_map.json":
        upload_files.append(file_name)

api = HfApi()
if branch_name != "main":
    try:
        api.create_branch(
            repo_id=repo_name,
            repo_type="model",
            branch=branch_name,
        )
    except Exception:
        print(f"branch {branch_name} already exists, try again...")
print(f"to upload: {upload_files}")
for file in upload_files:
    print(f"Uploading {file} to branch {branch_name}...")
    api.upload_file(
        path_or_fileobj=os.path.join(converted_ckpt, file),
        path_in_repo=file,
        repo_id=repo_name,
        repo_type="model",
        commit_message=f"Upload {file}",
        revision=branch_name,
    )
    print(f"Successfully uploaded {file} !")
