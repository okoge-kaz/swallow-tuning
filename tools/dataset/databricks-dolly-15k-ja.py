import argparse
import os
from tqdm import tqdm
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

    instruction_data = []

    for conversation in tqdm(jsonl_data):
        instruction_conversation: dict = {
            "input": [],
        }

        if len(conversation["context"]) > 0:
            instruction_conversation["input"].append(
                {
                    "role": "user",
                    "text": conversation["instruction"] + "\n\n" + conversation["context"]
                }
            )
        else:
            instruction_conversation["input"].append(
                {
                    "role": "user",
                    "text": conversation["instruction"]
                }
            )
        instruction_conversation["output"] = conversation["response"]
        instruction_data.append(instruction_conversation)

    print(f"\n\nlen(instruction_data)={len(instruction_data)}")

    # save
    with open(args.output, "w") as f:
        for instruction_pair in instruction_data:
            f.write(json.dumps(instruction_pair, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
