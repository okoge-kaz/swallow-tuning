import argparse
from typing import Any
from tqdm import tqdm  # type: ignore
import json


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, nargs='+', required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = arg_parse()

    jsonl_data: list = []
    for file in args.input:
        with open(file, "r") as f:
            try:
                jsonl_data.extend(json.load(f))
            except Exception as e:
                print(f"file name={file}, error={e}")

    instruction_data: list[dict[str, Any]] = []
    for conversation in tqdm(jsonl_data):
        instruction_conversation: dict = {
            "input": [],
        }

        instruction_conversation["input"].append(
            {
                "role": "user",
                "text": conversation["text"]
            }
        )

        instruction_conversation["output"] = conversation["output"]

        instruction_conversation["id"] = conversation["ID"]

        instruction_data.append(instruction_conversation)

    print(f"\n\nlen(instruction_data)={len(instruction_data)}")

    # save
    with open(args.output, "w") as f:
        for instruction_pair in instruction_data:
            f.write(json.dumps(instruction_pair, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
