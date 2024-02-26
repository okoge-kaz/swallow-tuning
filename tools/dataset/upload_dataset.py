import argparse
from datasets import load_dataset


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--upload", type=str)

    return parser.parse_args()


args = arg_parse()

dataset = load_dataset("json", data_files=args.input)
dataset.push_to_hub(args.upload)  # type: ignore
