import argparse
from typing import Any
from tqdm import tqdm  # type: ignore
import json


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = arg_parse()

    jsonl_data: list = []
    with open(args.input, "r") as f:
        for line in f:
            jsonl_data.append(json.loads(line))

    with open(args.output, 'w', encoding='utf-8') as file:
        output_data: dict = {"version": "v1.0", "data": jsonl_data}
        json.dump(output_data, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
