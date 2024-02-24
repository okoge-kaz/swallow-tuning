import argparse
from typing import Any
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

    instruction_data: list[dict[str, Any]] = []
    for conversations in tqdm(jsonl_data):
        instruction_conversation: dict = {
            "input": [],
        }

        for conversation in conversations["conversations"]:

            if conversation["from"] == "human":
                instruction_conversation["input"].append(
                    {
                        "role": "user",
                        "text": conversation["value"]
                    }
                )
            elif conversation["from"] == "gpt":
                instruction_conversation["output"] = conversation["value"]
                instruction_data.append(instruction_conversation)

                instruction_conversation = {
                    "input": instruction_conversation["input"].copy(),
                }
                instruction_conversation["input"].append(
                    {
                        "role": "assistant",
                        "text": conversation["value"]
                    }
                )
            else:
                print(f"invalid conversation={conversation}")

    print(f"\n\nlen(instruction_data)={len(instruction_data)}")

    # save
    with open(args.output, "w") as f:
        for instruction_pair in instruction_data:
            f.write(json.dumps(instruction_pair, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
